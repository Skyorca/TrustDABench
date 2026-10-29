# coding=utf-8
"""
子进程隔离安全沙箱
✅ 支持真正的超时强制终止，C级阻塞也能杀
✅ 跨代码执行保留变量状态，符合多轮推理需求
✅ Windows/Linux双平台兼容
✅ 进程级隔离，代码崩溃/OOM不影响主进程
"""
import ast
import re
import pandas as pd
import sys
import os
import queue
import signal
import time
import traceback
from dataclasses import dataclass
from multiprocessing import Process, Queue, Lock
from pathlib import Path
from dataclasses_json import DataClassJsonMixin

def trim_long_string(string, threshold=5100, k=2500):
    if len(string) > threshold:
        first_k_chars = string[:k]
        last_k_chars = string[-k:]
        truncated_len = len(string) - 2 * k
        return f"{first_k_chars}\n ... [{truncated_len} characters truncated] ... \n{last_k_chars}"
    else:
        return string

@dataclass
class ExecutionResult(DataClassJsonMixin):
    term_out: list[str]
    exec_time: float
    exc_type: str | None
    exc_info: dict | None = None
    exc_stack: list[tuple] | None = None

def exception_summary(e, working_dir, exec_file_name, format_tb_ipython):
    if format_tb_ipython:
        import IPython.core.ultratb
        tb = IPython.core.ultratb.VerboseTB(tb_offset=1, color_scheme="NoColor")
        tb_str = str(tb.text(*sys.exc_info()))
    else:
        tb_lines = traceback.format_exception(e)
        tb_str = "".join(
            [l for l in tb_lines if "ml-master/" not in l and "importlib" not in l]
        )
    tb_str = tb_str.replace(str(working_dir / exec_file_name), exec_file_name)
    exc_info = {}
    if hasattr(e, "args"):
        exc_info["args"] = [str(i) for i in e.args]
    for att in ["name", "msg", "obj"]:
        if hasattr(e, att):
            exc_info[att] = str(getattr(e, att))
    tb = traceback.extract_tb(e.__traceback__)
    exc_stack = [(t.filename, t.lineno, t.name, t.line) for t in tb]
    return tb_str, e.__class__.__name__, exc_info, exc_stack

class RedirectQueue:
    def __init__(self, queue):
        self.queue = queue
    def write(self, msg:str):
        self.queue.put(msg)
    def flush(self):
        pass

class Interpreter:
    def __init__(
        self,
        timeout: int = 1200,
        format_tb_ipython: bool = False,
        agent_file_name: str = "runfile.py",
        max_parallel_run: int = 1,
        start_cpu_id: int = 0,
        cpu_number: int = 1,
    ):
        self.timeout = timeout
        self.format_tb_ipython = format_tb_ipython
        self.max_parallel_run = max_parallel_run
        self.agent_file_name = [f"runfile_{i}.py" for i in range(self.max_parallel_run)]
        self.process: list[Process] = [None] * self.max_parallel_run
        self.current_parallel_run = 0
        self.status_map = [0] * self.max_parallel_run
        self.code_inq = [None] * self.max_parallel_run
        self.result_outq = [None] * self.max_parallel_run
        self.event_outq = [None] * self.max_parallel_run
        self.start_cpu_id = start_cpu_id
        self.cpu_number = cpu_number
        if self.cpu_number < self.max_parallel_run:
            raise ValueError("The maximum level of parallelism exceeds the number of allocated CPU cores")
        self.lock = Lock()
        # 工作目录：通过Queue传递给子进程，不依赖主进程属性
        self._working_dir = None

    def set_working_dir(self, working_dir):
        """设置工作目录（在主进程调用，子进程通过Queue获取）"""
        self._working_dir = str(Path(working_dir).resolve())

    def check_current_status(self):
        """检查是否有空闲进程槽位（只读，不持锁，结果仅供参考）"""
        return self.current_parallel_run < self.max_parallel_run

    def _run_session(
        self, code_inq: Queue, result_outq: Queue, event_outq: Queue, process_id: int
    ) -> None:
        """子进程主循环：从Queue接收代码和工作目录，执行代码"""
        # 启动时静默 Python 警告，减少 DeprecationWarning 等噪音输出
        try:
            import shutup
            shutup.mute_warnings()
        except ImportError:
            pass  # shutup 未安装则跳过，不影响功能

        global_scope: dict = {}
        global_scope["__name__"] = "__main__"

        while True:
            # 从Queue接收 (code, id, working_dir) 三元组
            item = code_inq.get()
            code, _task_id, working_dir = item

            # 每次执行前都切换到正确的工作目录
            os.chdir(working_dir)
            if working_dir not in sys.path:
                sys.path.append(working_dir)

            # 重定向stdout/stderr到Queue（在chdir之后，避免路径问题）
            sys.stdout = sys.stderr = RedirectQueue(result_outq)

            # Windows平台跳过CPU亲和性设置
            if sys.platform == "linux":
                cpu_number_per_session = int(self.cpu_number / self.max_parallel_run)
                start_cpu_id_session = self.start_cpu_id + process_id * cpu_number_per_session
                cpu_set = set()
                for i in range(start_cpu_id_session, start_cpu_id_session + cpu_number_per_session):
                    cpu_set.add(i)
                pre_code = "import os\nos.sched_setaffinity(0, {cpu_set})\n".format(cpu_set=cpu_set)
                code = pre_code + code

            agent_file = self.agent_file_name[process_id]
            with open(agent_file, "w", encoding="utf-8") as f:
                f.write(code)
            event_outq.put(("state:ready",))
            try:
                exec(compile(code, agent_file, "exec"), global_scope)
            except BaseException as e:
                tb_str, e_cls_name, exc_info, exc_stack = exception_summary(
                    e,
                    Path(working_dir),
                    agent_file,
                    self.format_tb_ipython,
                )
                result_outq.put(tb_str)

                if e_cls_name == "KeyboardInterrupt":
                    e_cls_name = "TimeoutError"
                event_outq.put(("state:finished", e_cls_name, exc_info, exc_stack))
            else:
                event_outq.put(("state:finished", None, None, None))

            # 清理临时文件
            try:
                if os.path.exists(agent_file):
                    os.remove(agent_file)
            except:
                pass
            result_outq.put("<|EOF|>")

    def create_process(self, process_id) -> None:
        self.code_inq[process_id] = Queue()
        self.result_outq[process_id] = Queue()
        self.event_outq[process_id] = Queue()
        self.process[process_id] = Process(
            target=self._run_session,
            args=(self.code_inq[process_id], self.result_outq[process_id], self.event_outq[process_id], process_id),
        )
        # 必须设置 daemon=False，否则如果父进程是 Pool worker（daemon进程），
        # 创建子进程时会报 "daemonic processes are not allowed to have children"
        self.process[process_id].daemon = False
        self.process[process_id].start()

    def cleanup_session(self, process_id):
        if process_id == -1:
            # 清理所有进程
            for pid in range(self.max_parallel_run):
                if self.process[pid] is None:
                    continue
                try:
                    self.process[pid].terminate()
                    self.process[pid].join(timeout=2)
                    if self.process[pid].exitcode is None:
                        self.process[pid].kill()
                        self.process[pid].join()
                    self.process[pid].close()
                    self.process[pid] = None
                except:
                    pass
        else:
            # 清理指定进程
            if self.process[process_id] is None:
                return
            try:
                self.process[process_id].terminate()
                self.process[process_id].join(timeout=2)
                if self.process[process_id].exitcode is None:
                    self.process[process_id].kill()
                    self.process[process_id].join()
                self.process[process_id].close()
                self.process[process_id] = None
            except:
                pass

    def _drain_queue(self, process_id):
        """安全地排空输出队列，避免超时kill后get()死锁"""
        output = []
        try:
            while True:
                res = self.result_outq[process_id].get(timeout=1)
                output.append(str(res))
                if str(res) == "<|EOF|>":
                    break
        except:
            # 子进程被kill后不会再写EOF，直接返回已收集的内容
            pass
        return output

    def run(self, code: str, id="default", reset_session=False):
        assert self._working_dir is not None, "Please call set_working_dir() first"

        process_id = None
        with self.lock:
            self.current_parallel_run += 1
            for idx in range(self.max_parallel_run):
                if self.status_map[idx] == 0:
                    self.status_map[idx] = 1
                    process_id = idx
                    break
                elif idx == self.max_parallel_run - 1:
                    # 没有空闲进程，回滚计数
                    self.current_parallel_run -= 1
                    raise ValueError("reach max process parallel number")

        # 用 try/finally 确保 status_map 和 current_parallel_run 一定会被重置，
        # 防止任何异常路径导致状态泄漏，后续调用全部卡死
        process_killed = False  # 标记子进程是否已被 cleanup_session 清理
        try:
            try:
                if reset_session or self.process[process_id] is None:
                    if self.process[process_id] is not None:
                        try:
                            self.cleanup_session(process_id=process_id)
                        except:
                            pass
                    self.create_process(process_id=process_id)

                if not self.process[process_id].is_alive():
                    # 子进程已死但状态未清理，重新创建
                    self.cleanup_session(process_id=process_id)
                    self.create_process(process_id=process_id)
            except Exception:
                # 创建子进程失败时，必须重置状态，否则后续调用全部卡死
                raise

            # 通过Queue传递 (code, id, working_dir) 给子进程
            self.code_inq[process_id].put((code, id, self._working_dir))
            try:
                state = self.event_outq[process_id].get(timeout=10)
            except queue.Empty:
                msg = "REPL child process failed to start execution"
                queue_dump = ""
                while not self.result_outq[process_id].empty():
                    queue_dump = self.result_outq[process_id].get()
                self.cleanup_session(process_id=process_id)
                process_killed = True
                return ExecutionResult(
                    term_out=[msg, str(queue_dump)],
                    exec_time=0,
                    exc_type="RuntimeError",
                    exc_info={},
                    exc_stack=[],
                )

            assert state[0] == "state:ready", state
            start_time = time.time()
            child_in_overtime = False

            while True:
                try:
                    state = self.event_outq[process_id].get(timeout=1)
                    assert state[0] == "state:finished", state
                    exec_time = time.time() - start_time
                    break
                except queue.Empty:
                    if not child_in_overtime and not self.process[process_id].is_alive():
                        msg = "REPL child process died unexpectedly"
                        # 子进程已死，用安全方式排空队列
                        output = self._drain_queue(process_id)
                        self.cleanup_session(process_id=process_id)
                        process_killed = True
                        return ExecutionResult(
                            term_out=[msg] + output,
                            exec_time=0,
                            exc_type="RuntimeError",
                            exc_info={},
                            exc_stack=[],
                        )

                    if self.timeout is None:
                        continue
                    running_time = time.time() - start_time
                    if running_time > self.timeout:
                        # 超时：先发SIGINT，给子进程60秒优雅退出
                        try:
                            os.kill(self.process[process_id].pid, signal.SIGINT)
                        except:
                            pass
                        child_in_overtime = True
                        if running_time > self.timeout + 60:
                            # 60秒后还没退出，强制kill
                            self.cleanup_session(process_id=process_id)
                            process_killed = True
                            state = (None, "TimeoutError", {}, [])
                            exec_time = self.timeout
                            break

            # 读取输出：超时kill后用安全排空方式，避免死锁
            # 注意：cleanup_session 后 self.process[process_id] 为 None，
            # 不能再调用 self.process[process_id].is_alive()，用 process_killed 标记判断
            if process_killed:
                output = self._drain_queue(process_id)
            else:
                output = []
                while not self.result_outq[process_id].empty() or not output or output[-1] != "<|EOF|>":
                    try:
                        res = self.result_outq[process_id].get(timeout=5)
                        output.append(str(res))
                    except queue.Empty:
                        break

            # 移除EOF标记
            if output and output[-1] == "<|EOF|>":
                output.pop()

            e_cls_name, exc_info, exc_stack = state[1:]

            if e_cls_name == "TimeoutError":
                output.append(f"TimeoutError: Execution exceeded the time limit of {self.timeout} seconds")
            else:
                output.append(f"Execution time: {exec_time:.1f} seconds (time limit is {self.timeout} seconds)")
            return ExecutionResult(
                term_out=output,
                exec_time=exec_time,
                exc_type=e_cls_name,
                exc_info=exc_info,
                exc_stack=exc_stack,
            )
        finally:
            # 无论走哪条路径（正常结束、超时kill、异常），都确保重置状态
            self.current_parallel_run -= 1
            self.status_map[process_id] = 0

# 原有函数保留，保持接口完全兼容
def parse_code_from_string(input_string):
    triple_backtick_pattern = r"```(\w*\s*)?(.*?)```"
    match = re.search(triple_backtick_pattern, input_string, flags=re.DOTALL | re.IGNORECASE)
    if match:
        return match.group(2).strip()
    single_backtick_pattern = r"`(.*?)`"
    match = re.search(single_backtick_pattern, input_string, flags=re.DOTALL)
    if match:
        return match.group(1).strip()
    return input_string.strip()

def object_to_string(obj, command):
    if isinstance(obj, str):
        return obj
    elif isinstance(obj, pd.DataFrame):
        if len(obj) == 0:
            return 'Empty DataFrame'
    elif command == 'df.columns':
        obj = obj.tolist()
        if len(obj) > 20:
            return str(obj[:10])[:-1] + ', ..., ' + str(obj[-10:])[1:]
    return str(obj)

# 原有MyBox类保持接口完全兼容
class MyBox:
    def __init__(self, config=None, working_dir=None, timeout=1200):
        self.config = config
        # 初始化子进程沙箱，单步超时由评测配置控制
        self.interpreter = Interpreter(timeout=timeout, max_parallel_run=1)
        self.memory = {}  # 兼容原有字段，实际变量存储在子进程中
        # 如果创建时就知道工作目录，直接设置
        if working_dir is not None:
            self.interpreter.set_working_dir(working_dir)

    def run(self, code, custom_locals={}, custom_globals={}):
        # custom_locals/custom_globals 保留参数为了接口兼容，子进程沙箱内部通过global_scope管理变量
        result = self.interpreter.run(code)
        # Interpreter.run() 返回 ExecutionResult，调用方期望字符串，在此做转换
        if isinstance(result, ExecutionResult):
            return "\n".join(result.term_out)
        return result
