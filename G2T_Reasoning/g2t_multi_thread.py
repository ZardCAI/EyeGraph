import numpy as np
import threading
from tqdm import tqdm
from queue import Queue
from G2T_Reasoning.G2T_reasoning import PureDFSReasoner
import pandas as pd
import os
import torch
import time
from neo4j import GraphDatabase
from kg_searcher import FactSearcher

# 配置参数
DATA_PATH = 'caption_test.csv'
ROOT_PATH = 'CFP'
NUM_WORKERS = 8
DEVICES = [0, 1, 2, 3, 4, 5, 6, 7]  # 使用的GPU设备编号
SAVE_INITIAL = 'initial_test_50.json'
SAVE_T2G = 't2g_test_50.json'

fact_searcher = FactSearcher(nums=5)

# 准备数据
def prepare_data():
    data = pd.read_csv(DATA_PATH)
    data['Path'] = data.apply(lambda row: os.path.join(ROOT_PATH, f"{row['Img ID']}.png"), axis=1)
    return data[data['Img ID'].str.contains('_1')].reset_index()

# 同步管理器
class SyncManager:
    def __init__(self, total):
        self.barrier = threading.Barrier(total)
        self.exception = None
        self.lock = threading.Lock()

# 处理线程
class ProcessingThread(threading.Thread):
    def __init__(self, chunk, device, sync_manager, progress_queue):
        super().__init__()
        self.chunk = chunk
        self.device = device
        self.sync = sync_manager
        self.progress_queue = progress_queue
        self.reasoner = None
        self.results = []

    def run(self):
        try:
            # 初始化阶段
            print(f'Device {self.device}: len(chunk) = {len(self.chunk)}')
            torch.cuda.set_device(self.device)
            self.reasoner = PureDFSReasoner(device=self.device, fact_searcher=fact_searcher)
            print(f"Device {self.device}: Model loaded")

            # 等待所有线程完成模型加载
            self.sync.barrier.wait()

            # 数据处理阶段
            for _, row in self.chunk.iterrows():
                img_path = row['Path']
                img_id = row['Img ID']
                gt_report = row['Caption']
                
                initial_report, final_report = self.reasoner.reasoning(img_path)
                self.results.append((img_id, gt_report, initial_report, final_report))
                self.progress_queue.put(1)

        except Exception as e:
            with self.sync.lock:
                self.sync.exception = e
                print(f"Error on device {self.device}: {str(e)}")
            self.progress_queue.put(None)
            raise
        finally:
            self.progress_queue.put(None)

# 主流程
def main():
    # 准备数据
    test_data = prepare_data()
    data_splits = np.array_split(test_data, NUM_WORKERS)
    
    # 创建共享队列和同步管理器
    progress_queue = Queue()
    sync_manager = SyncManager(NUM_WORKERS)
    results = []
    results_lock = threading.Lock()

    # 创建并启动线程
    threads = []
    for i in range(NUM_WORKERS):
        thread = ProcessingThread(
            chunk=data_splits[i],
            device=DEVICES[i],
            sync_manager=sync_manager,
            progress_queue=progress_queue
        )
        thread.start()
        threads.append(thread)

    # 进度监控
    completed_threads = 0
    with tqdm(total=len(test_data), desc="Processing") as pbar:
        while completed_threads < NUM_WORKERS:
            if sync_manager.exception:
                print(f"Critical error occurred: {sync_manager.exception}")
                for t in threads:
                    t.join(timeout=1)
                raise RuntimeError("Aborting due to worker error")

            item = progress_queue.get()
            if item is None:
                completed_threads += 1
            else:
                pbar.update(1)
            progress_queue.task_done()

    # 收集结果
    for thread in threads:
        with results_lock:
            results.extend(thread.results)

    # 保存结果
    pred_initial_dict = {}
    pred_t2g_dict = {}
    true_dict = {}
    
    for img_id, gt_report, initial_report, final_report in results:
        pred_initial_dict[img_id] = initial_report
        pred_t2g_dict[img_id] = final_report
        true_dict[img_id] = gt_report

    for path, data in [(SAVE_INITIAL, pred_initial_dict), (SAVE_T2G, pred_t2g_dict)]:
        with open(path, 'w') as f:
            json.dump({
                'pred_dict': data,
                'true_dict': true_dict
            }, f)

    print(f"Processing completed. Total results: {len(results)}")

if __name__ == "__main__":
    import json
    main()