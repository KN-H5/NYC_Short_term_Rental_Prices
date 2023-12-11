#!/usr/bin/env python
"""
Performs basic cleaning on the data and save the results in Weights & Biases
"""
import argparse
import logging
import os
import wandb
import pandas as pd


logging.basicConfig(level=logging.INFO, format="%(asctime)-15s %(message)s")
logger = logging.getLogger()


def go(args):
    """
    Run basic cleaning on the data and save the results in Weights
    """
    run = wandb.init(job_type="basic_cleaning")
    run.config.update(args)

    # Download input artifact. This will also log that this script is using this
    # particular version of the artifact
    artifact_local_path = run.use_artifact(args.input_artifact).file()

    df = pd.read_csv(artifact_local_path, index_col="id")
    indexes = df['price'].between(args.min_price, args.max_price)
    df = df[indexes].copy()
    logger.info(
        f"Dataset price outliers removal outside range: {args.min_price}-{args.max_price}")

    # Convert last_review to datetime
    df['last_review'] = pd.to_datetime(df['last_review'])
    logger.info("Converted last_review to datetime")

    idx = df['longitude'].between(-74.25, -73.50) & df['latitude'].between(40.5, 41.2)
    df = df[idx].copy()

    df.to_csv("clean_sample.csv")
    logger.info("Saved dataframe to local")

    artifact = wandb.Artifact(
        args.output_artifact,
        type=args.output_type,
        description=args.output_description,
    )
    artifact.add_file("clean_sample.csv")
    run.log_artifact(artifact)

    artifact.wait()
    logger.info("Uploaded clean dataset to wandb!")


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="This steps cleans the data")

    parser.add_argument(
        "--input_artifact",
        type=str,
        help="The input artifact",
        required=True
    )

    parser.add_argument(
        "--output_artifact",
        type=str,
        help="The name for the output artifact",
        required=True
    )

    parser.add_argument(
        "--output_type",
        type=str,
        help="The type for the output artifact",
        required=True
    )

    parser.add_argument(
        "--output_description",
        type=str,
        help="A description for the output artifact",
        required=True
    )

    parser.add_argument(
        "--min_price",
        type=float,
        help="The minimum price to consider",
        required=True
    )

    parser.add_argument(
        "--max_price",
        type=float,
        help="The maximum price to consider",
        required=True
    )

    args = parser.parse_args()

    go(args)