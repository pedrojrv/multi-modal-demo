import lancedb
import pandas as pd
import pyarrow as pa

from src.constants import VECTORDB_PATH


class DemoVectorDB:
    def __init__(self, path):
        self.path = path
        self.db = None

    async def connect(self):
        self.db = await lancedb.connect_async(self.path)

    def create_image_table(self):
        schema = pa.schema(
            [
                pa.field("vector", pa.list_(pa.float32(), list_size=2)),
                pa.field("image", pa.binary()),
                pa.field("caption", pa.string()),
            ]
        )
        self.db.create_table("image_table", schema=schema)

db = lancedb.connect(VECTORDB_PATH)

async_db = await lancedb.connect_async(VECTORDB_PATH)


data = [
    {"vector": [3.1, 4.1], "item": "foo", "price": 10.0},
    {"vector": [5.9, 26.5], "item": "bar", "price": 20.0},
]

# Synchronous client
tbl = db.create_table("my_table", data=data)
# Asynchronous client
async_tbl = await async_db.create_table("my_table2", data=data)


schema = pa.schema([pa.field("vector", pa.list_(pa.float32(), list_size=2))])
# Synchronous client
tbl = db.create_table("empty_table", schema=schema)
# Asynchronous client
async_tbl = await async_db.create_table("empty_table2", schema=schema)

# Synchronous client
tbl = db.open_table("my_table")
# Asynchronous client
async_tbl = await async_db.open_table("my_table2")

# Synchronous client
print(db.table_names())
# Asynchronous client
print(await async_db.table_names())

# Option 1: Add a list of dicts to a table
data = [
    {"vector": [1.3, 1.4], "item": "fizz", "price": 100.0},
    {"vector": [9.5, 56.2], "item": "buzz", "price": 200.0},
]
tbl.add(data)

# Option 2: Add a pandas DataFrame to a table
df = pd.DataFrame(data)
tbl.add(data)
# Asynchronous client
await async_tbl.add(data)


# Synchronous client
tbl.search([100, 100]).limit(2).to_pandas()
# Asynchronous client
await async_tbl.vector_search([100, 100]).limit(2).to_pandas()


# Synchronous client
tbl.create_index(num_sub_vectors=1)
# Asynchronous client (must specify column to index)
await async_tbl.create_index("vector")


# Synchronous client
tbl.delete('item = "fizz"')
# Asynchronous client
await async_tbl.delete('item = "fizz"')

# Synchronous client
db.drop_table("my_table")
# Asynchronous client
await async_db.drop_table("my_table2")



from lancedb.pydantic import LanceModel, Vector
from lancedb.embeddings import get_registry

db = lancedb.connect("/tmp/db")
func = get_registry().get("openai").create(name="text-embedding-ada-002")

class Words(LanceModel):
    text: str = func.SourceField()
    vector: Vector(func.ndims()) = func.VectorField()

table = db.create_table("words", schema=Words, mode="overwrite")
table.add([{"text": "hello world"}, {"text": "goodbye world"}])

query = "greetings"
actual = table.search(query).limit(1).to_pydantic(Words)[0]
print(actual.text)