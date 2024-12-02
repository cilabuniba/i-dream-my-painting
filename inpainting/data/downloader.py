from langchain_community.graphs import Neo4jGraph
import requests
from PIL import Image
from pathlib import Path
import concurrent.futures
from tqdm import tqdm
import pandas as pd
import click


@click.group()
def cli():
    pass


def download_and_save_images_wikiart(output_dir: Path | str) -> None:
    """Download and save images from WikiArt.

    Args:
        output_dir (Path | str): The directory to save the images.
    """
    graph = Neo4jGraph(
        url="bolt://localhost:7687", username="neo4j", password="artgraph"
    )

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # list all already present images
    existing_images = set([img.name for img in output_dir.glob("*.jpg")])

    modifiers = [
        "HalfHD",
        "Large",
        "PinterestLarge",
        "PortraitSmall",
        "PinterestSmall",
    ]

    query = """
    MATCH (a:Artwork)
    WHERE a.image_url IS NOT NULL
    RETURN a.image_url AS image_url, a.name AS name
    """
    result = graph.query(query)

    # check if wikiart_images.csv already exists
    if (output_dir / "wikiart_images.csv").exists():
        dw_df = pd.read_csv(output_dir / "wikiart_images.csv")
        dw_df = dw_df.to_dict()
        existing_images.update(dw_df["file_name"].values)
    else:
        dw_df = {"file_name": [], "url": [], "modifier": []}

    # Define a function that handles downloading and saving the image
    def download_image(record):
        image_url = record["image_url"]
        name = record["name"]

        # If a modifier is already present in the URL, remove it
        modifier_idx = image_url.find("!")
        if modifier_idx != -1:
            image_url = image_url[:modifier_idx]

        # If an extension is already present in the URL, use it otherwise it's a jpg
        ext = image_url.split(".")[-1]

        # Skip if the image is already downloaded
        if name in existing_images:
            return f"Already downloaded"

        # Try the image URL with different modifiers
        for modifier in modifiers:
            modified_url = image_url + f"!{modifier}.{ext}"
            response = requests.get(modified_url, stream=True, timeout=5)
            if response.status_code == 200:
                break
        else:
            # If no modifier results in a valid image, return error
            return f"Error downloading {image_url}: {response.status_code}"
        try:
            # Process and save the image
            image = Image.open(response.raw)
            image = image.convert("RGB")
            # If name contains an ext different from jpg, change it
            if name.split(".")[-1] != "jpg":
                name = name.split(".")[0] + ".jpg"
            image.save(output_dir / f"{name}")
            with open(output_dir / f"{name}.txt", "w") as f:
                # write url and modifier to a txt file
                f.write(f"{image_url}\n{modifier}")
            return (name, image_url, modifier)
        except Exception as e:
            return f"Error processing {name}: {e}"

    # Use ThreadPoolExecutor to download images in parallel
    with concurrent.futures.ThreadPoolExecutor() as executor:
        futures = [executor.submit(download_image, record) for record in result]

        # Show progress bar and gather results
        for future in tqdm(concurrent.futures.as_completed(futures), total=len(result)):
            if isinstance(future.result(), tuple):
                dw_df["file_name"].append(future.result()[0])
                dw_df["url"].append(future.result()[1])
                dw_df["modifier"].append(future.result()[2])

    # Save the download results to a CSV file
    dw_df = pd.DataFrame(dw_df)
    dw_df.to_csv(output_dir / "wikiart_images.csv", index=False)


@cli.command()
@click.option(
    "--output_dir",
    "-o",
    type=click.Path(exists=False),
    default="data/mm_inp_dataset/images",
)
def download_and_save_images_wikiart_v2(output_dir: Path | str) -> None:
    df = pd.read_csv("data/wikiart_images.csv")
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Define a function that handles downloading and saving the image
    def download_and_save_image(row):
        url = row["url"]
        file_name = row["file_name"]
        modifier = row["modifier"]

        # if file already exists, skip
        if (output_dir / file_name).exists():
            return

        ext = url.split(".")[-1]
        modified_url = url + f"!{modifier}.{ext}"
        try:
            img = Image.open(requests.get(modified_url, stream=True).raw).convert("RGB")
            img.save(output_dir / file_name)
        except Exception as e:
            print(f"Error downloading {file_name}: {e}")

    # Define the number of threads
    num_threads = 16

    with concurrent.futures.ThreadPoolExecutor(max_workers=num_threads) as executor:
        list(
            tqdm(
                executor.map(
                    download_and_save_image, [row for _, row in df.iterrows()]
                ),
                total=len(df),
            )
        )


if __name__ == "__main__":
    cli()
