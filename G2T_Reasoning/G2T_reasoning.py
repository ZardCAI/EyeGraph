from vlm_responser import EyeGraphResponser
from scorer import Scorer
import os
from neo4j import GraphDatabase
from kg_searcher import FactSearcher

PROMPT_TEMPLATES = {
    "initial": "<image>\nAssume you are an ophthalmology expert, describe this fundus image.",
    "level_up": lambda entity: f"<image>\nAssume you are an ophthalmology expert, describe this fundus image, with attention to {entity}.",
    "final_generation": lambda key_points: f"<image>\nConsidering findings including {key_points}, describe this fundus image as an ophthalmology expert."
}

INVAILID_KEYWORDS = ['pole', 'skin', 'pulp', 'nose', 'dentition']

class PureDFSReasoner:
    def __init__(self, device, fact_searcher=None, max_depth=5, max_entities=10, sync=None):
        self.vlm_responser = EyeGraphResponser(device=device, sync=sync)
        self.fact_seacher = fact_searcher if fact_searcher else FactSearcher(nums=5)
        self.scorer = Scorer()

        self.max_depth = max_depth        # 最大递归深度
        self.max_entities = max_entities  # 最大实体数量
        
    def reasoning(self, img_path):
        self.collected_entities = []
        self.used_entities = set()  # 记录已使用的实体

        img_id = os.path.basename(img_path).split('.png')[0]
        f = open(f'/results/{img_id}.txt', 'w')
        initial_report = self.vlm_responser.reponse(PROMPT_TEMPLATES['initial'], img_path)
        initial_score = self.scorer._score_report(initial_report)
        f.write(initial_report + '\n')
        f.write(f'Initial Score: {initial_score}\n')
        
        # 启动DFS
        try:
            self._dfs(
                img_path,
                f,
                current_depth=0,
                parent_report=initial_report,
                parent_score=initial_score,
            )
        except Exception as e:
            f.write(f"Error during DFS: {e}")
            f.close()
            return initial_report, initial_report
         
        final_report = self._generate_report(img_path, f, initial_report)
        f.close()
        return initial_report, final_report

    def _dfs(self, img_path, f, current_depth, parent_report, parent_score):
        if current_depth >= self.max_depth or len(self.collected_entities) >= self.max_entities:
            return

        # 获取并筛选实体
        raw_facts = self.fact_seacher.search_fact(parent_report)  # 获取5个按相关性排序的实体
        facts = self._get_available_facts(raw_facts)  # 实际处理最多3个未使用的
        
        for entity in facts:
            # 标记为已使用（即使后续分数不提升也避免重复）
            # self.used_entities.add(entity)
            
            new_report = self.vlm_responser.reponse(
                PROMPT_TEMPLATES["level_up"](entity), 
                img_path
            )
            new_report = new_report.replace('\n', ' ')
            f.write('New Report: ' + new_report + '\n')
            new_score = self.scorer._score_report(new_report)
            
            f.write(f'Level{current_depth}: Fact:{entity}, Score:{new_score}\n')
            # print(f'Level{current_depth}: Fact:{entity}, Score:{new_score}\n')
            
            if new_score > parent_score:
                self.collected_entities.append(entity)
                self._dfs(
                    img_path,
                    f,
                    current_depth=current_depth + 1,
                    parent_report=new_report,
                    parent_score=new_score,
                )

    def _get_available_facts(self, facts):
        """从5个候选实体中筛选最多前3个未使用的"""
        available = []
        for entity in facts:

            keywords_check = False
            for keywords in INVAILID_KEYWORDS:
                if keywords in entity:
                    keywords_check = True
                    break
            if keywords_check:
                continue

            if entity not in self.used_entities:
                available.append(entity)
                self.used_entities.add(entity)
                if len(available) >= 3:  # 最多取3个可用实体
                    break
        return available  # 确保不超过3个

    def _generate_report(self, img_path, f, initial_report):
        if len(self.collected_entities) == 0:
            return initial_report
        
        # f.write(len(self.collected_entities))
        f.write(", ".join(self.collected_entities))
        f.write('\n')
        final_report = self.vlm_responser.reponse(PROMPT_TEMPLATES['final_generation'](', '.join(self.collected_entities)), img_path)
        f.write(final_report)
        return final_report
    

# if __name__ == '__main__':
# # print(1)
#     img_path = '315_1.png'
#     t2g = PureDFSReasoner(device=4)
#     report = t2g.reasoning(img_path)
#     print(report)
