import lancedb
import logging
import pandas as pd
import pyarrow as pa


logger = logging.getLogger(__name__)


class AsyncDemoVectorDB:
    def __init__(
            self,
            path_for_vectordb: str,
            txt_emb_fn: callable = None,
            img_emb_fn: callable = None,
            img_to_txt_fn: callable = None,
    ):
        self.path = path_for_vectordb
        self.txt_emb_fn = txt_emb_fn
        self.img_emb_fn = img_emb_fn
        self.img_to_txt_fn = img_to_txt_fn
        self.db = None
        self.table_names = None
        self.image_table = None
        self.text_table = None

    async def aconnect(self):
        self.db = await lancedb.connect_async(self.path)
        self.table_names = await self.db.table_names()
        logger.info(f"Connected to VectorDB at {self.path}")
        logger.info(f"Names of tables in the database: {self.table_names}")

    async def initialize(self):
        if "image_table" not in self.table_names:
            await self.create_image_table()
        if "text_table" not in self.table_names:
            await self.create_text_table()
        self.image_table = await self.db.open_table("image_table")
        self.text_table = await self.db.open_table("text_table")

    async def create_image_table(self):
        schema = pa.schema(
            [
                pa.field("vector_img", pa.list_(pa.float32(), list_size=768)),
                pa.field("vector_txt", pa.list_(pa.float32(), list_size=384)),
                pa.field("text", pa.string()),
                pa.field("image_name", pa.string()),
                pa.field("image", pa.binary()),
            ]
        )
        await self.db.create_table("image_table", schema=schema)

    async def create_text_table(self):
        schema = pa.schema(
            [
                pa.field("vector", pa.list_(pa.float32(), list_size=384)),
                pa.field("text", pa.string()),
            ]
        )
        await self.db.create_table("text_table", schema=schema)

    async def add_images(self, images) -> None:
        if not isinstance(images, list):
            images = [images]

        vectors_img = self.img_emb_fn(images)
        texts = self.img_to_txt_fn(images)
        vector_txt = self.txt_emb_fn(texts)

        images_as_bytes = [open(img, "rb").read() for img in images]

        to_add = [
            {
                "vector_img": vec_img,
                "vector_txt": vec_txt,
                "text": txt,
                "image_name": str(img_name),
                "image": img
            } for vec_img, vec_txt, txt, img_name, img in zip(vectors_img, vector_txt, texts, images, images_as_bytes)
        ]

        logger.info(f"Adding {len(images)} images to the database.")
        await self.image_table.add(to_add)

    async def add_texts(self, texts) -> None:
        if not isinstance(texts, list):
            texts = [texts]

        vector = self.txt_emb_fn(texts)

        logger.info(f"Adding {len(texts)} texts to the database.")
        logger.info(f"Texts: {texts}")
        logger.info(f"Vectors: {vector}")

        # ensure
        to_add = [
            {"vector": vec, "text": txt} for vec, txt in zip(vector, texts)
        ]

        await self.text_table.add(to_add)

    async def search_images_by_image(self, image: str) -> pd.DataFrame:
        vector = self.img_emb_fn(image)[0]
        return await self.image_table.vector_search(vector).to_pandas()

    async def search_images_by_text(self, text: str) -> pd.DataFrame:
        vector = self.txt_emb_fn(text)[0]
        return await self.image_table.vector_search(vector).to_pandas()

    async def search_texts(self, text: str) -> pd.DataFrame:
        vector = self.txt_emb_fn(text)[0]
        return await self.text_table.vector_search(vector).to_pandas()

    async def cleanup(self) -> None:
        await self.db.drop_table("image_table")  # , ignore_missing=True)
        await self.db.drop_table("text_table")  # , ignore_missing=True)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    import os
    import asyncio
    from src.constants import VECTORDB_PATH
    from src.utils import get_random_image_path
    from src.inference.vit_base import ViTImageEmbeddings
    from src.inference.bge_small import HFBGESmallEnV15
    from src.inference.llama32_vision import GroqLlama32Vision

    vit_model = ViTImageEmbeddings()
    bge_model = HFBGESmallEnV15()
    llama_model = GroqLlama32Vision()

    vectordb = AsyncDemoVectorDB(
        path_for_vectordb=VECTORDB_PATH,
        txt_emb_fn=bge_model.invoke,
        img_emb_fn=vit_model.invoke,
        img_to_txt_fn=llama_model.caption_image,
    )

    random_image = str(get_random_image_path())

    async def main():
        await vectordb.aconnect()
        await vectordb.initialize()
        await vectordb.add_texts(["hello", "world", "goodbye"])
        print(await vectordb.search_texts("hi"))
        await vectordb.add_images(random_image)
        print(await vectordb.search_images_by_image(random_image))
        print(await vectordb.search_images_by_text("white dog"))
        # await vectordb.cleanup()

    try:
        asyncio.run(main())
    except Exception as e:
        logger.error(e)
        asyncio.run(vectordb.cleanup())
        raise e
