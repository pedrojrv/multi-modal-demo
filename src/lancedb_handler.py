import lancedb
import pandas as pd
import pyarrow as pa


class AsyncDemoVectorDB:
    def __init__(self, path: str):
        self.path = path
        self.db = None
        self.table_names = None
        self.image_table = None
        self.text_table = None

    async def aconnect(self):
        self.db = await lancedb.connect_async(self.path)
        self.table_names = await self.db.table_names()
        self.image_table = await self.db.open_table("image_table")
        self.text_table = await self.db.open_table("text_table")

    async def initialize(self):
        if "image_table" not in self.table_names:
            await self.create_image_table()
        if "text_table" not in self.table_names:
            await self.create_text_table()

    async def create_image_table(self):
        schema = pa.schema(
            [
                pa.field("vector_img", pa.list_(pa.float32(), list_size=2)),
                pa.field("vector_txt", pa.list_(pa.float32(), list_size=2)),
                pa.field("text", pa.string()),
                pa.field("image", pa.binary()),
            ]
        )
        await self.db.create_table("image_table", schema=schema)

    async def create_text_table(self):
        schema = pa.schema(
            [
                pa.field("vector", pa.list_(pa.float32(), list_size=2)),
                pa.field("text", pa.string()),
            ]
        )
        await self.db.create_table("text_table", schema=schema)

    async def add_image(self, image: bytes) -> None:
        await self.image_table.add([{"vector": vector, "text": text, "image": image}])

    async def add_text(self, text: str) -> None:
        await self.text_table.add([{"vector": vector, "text": text}])

    async def search_image(self, vector: list[int], search_by: str = 'image') -> pd.DataFrame:
        return await self.image_table.search(vector).to_pandas()

    async def search_text(self, vector: list[int]) -> pd.DataFrame:
        return await self.text_table.search(vector).to_pandas()

    async def cleanup(self) -> None:
        await self.db.drop_table("image_table", ignore_missing=True)
        await self.db.drop_table("text_table", ignore_missing=True)
