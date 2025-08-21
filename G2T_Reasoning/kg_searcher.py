from neo4j import GraphDatabase

def index_exists(tx, index_name):
        query = """
        SHOW INDEXES YIELD name
        WHERE name = $index_name
        RETURN name
        """
        result = tx.run(query, index_name=index_name)
        return result.single() is not None

def create_fulltext_index(tx):
    query = """
    CREATE FULLTEXT INDEX entityNameIndex FOR (n:Entity) ON EACH [n.name];
    """
    tx.run(query)

class FactSearcher:
    def __init__(self, nums):
        self.nums = nums
        self.driver = self.init_connection()
        self.create_index()

    def create_index(self):
        with self.driver.session() as session:
            index_name = "entityNameIndex"
            if not session.execute_read(index_exists, index_name):
                session.execute_write(create_fulltext_index)
                print(f"Full-Text 索引 '{index_name}' 已创建")
            else:
                print(f"Full-Text 索引 '{index_name}' 已存在，无需创建")

    def init_connection(self):
        NEO4J_URI = "xxxx"
        NEO4J_USER = "xxxx"
        NEO4J_PASSWORD = "xxxxx"
        driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD))
        return driver

    def search_fact(self, query_report):
        def find_similar_entities(tx, query_string):
            query = """
            CALL db.index.fulltext.queryNodes('entityNameIndex', $query_string + '~') 
            YIELD node, score 
            RETURN node.name AS entity_name, score 
            ORDER BY score DESC
            """
            result = tx.run(query, query_string=query_string)
            return [record for record in result]

        with self.driver.session() as session:
            result = session.execute_read(find_similar_entities, query_report)
            
            if result:
                similarity_threshold = 2
                top_entities = [record for record in result if record["score"] >= similarity_threshold][:self.nums]
                entity_names = [record['entity_name'] for record in top_entities]
                # print(f"Top matches for query '{query_report}' with score >= {similarity_threshold}: {entity_names}")
                return entity_names
            else:
                # print(f"No matching entities found for query: {query_report}")
                return None
            
    def close_connection(self):
        self.driver.close()
