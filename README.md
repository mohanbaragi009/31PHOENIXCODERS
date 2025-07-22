Find My Fund Using LLM

Loaded 3 separate datasets in JSON Format:
Mutual Fund Master Data (mutual_funds.json)
Fund Holdings Data (fund_holdings.json)
Stock Metadata (stocks.json)
Converted JSON → DataFrames using pandas.read_json() and json_normalize()
Ensured data structure is flat and accessible for modeling
Merged datasets using:fund_id to link mutual funds ↔ holdingsstock_id to link holdings ↔ stock metadataCreated a consolidated view of each fund with its name, sector, category, holdings, AUM, and other metadata  To build a lightweight, intelligent search system that can accurately map a user's vague or fuzzy query to the correct Indian mutual fund, stock, or ETF.
To go beyond traditional string matching by using a Small Language Model (SLM) and semantic embeddings to understand user intent
To leverage metadata (fund house, category, sector, AUM, holdings) to:
To leverage metadata (fund house, category, sector, AUM, holdings) to:
Improve result precision
Disambiguate similar-sounding funds  Generate a description for the GitHub README.md

Certainly! Here’s a README.md description for your project, capturing its purpose, the data pipeline, and the semantic search capability:

# Intelligent Indian Mutual Fund & Stock Search System

## Overview

This project provides a *lightweight, intelligent search system* that enables users to accurately find and identify Indian mutual funds, stocks, and ETFs—even from vague or imperfect queries. Unlike traditional string-matching search, this system leverages *semantic embeddings* and a *Small Language Model (SLM)* to understand user intent and context, delivering highly relevant results and resolving ambiguity across similar-sounding fund or stock names.

==*Key Features *==

- *Semantic Search*: Uses advanced language models to interpret and map fuzzy or partial user queries to the most relevant financial entities.
- *Rich Metadata Utilization*: Improves search precision by leveraging comprehensive metadata, such as:
  - Fund house
  - Mutual fund category
  - Sector exposure
  - Assets Under Management (AUM)
  - Top holdings
  - Stock information (industry, sector, ticker, etc.)
- *Disambiguation*: Distinguishes between similarly named funds, stocks, or ETFs using metadata context and user intent.
- *Fast and Lightweight*: Designed for scalable performance and easy integration.

## Data Pipeline:

The system ingests and processes three primary data sources in JSON format:
- mutual_funds.json: Metadata for Indian mutual funds (fund_id, name, category, fund house, AUM, etc.)
- fund_holdings.json: Fund-constituent holdings mapping (fund_id, stock_id, percentage/allocation, etc.)
- stocks.json: Stock-level metadata (stock_id, ticker, sector, industry, company name, etc.)

*Processing Flow:*
1. Convert JSON files to Pandas DataFrames (read_json() + json_normalize()).
2. Flatten and normalize all data for consistent access and modeling.
3. Merge datasets:
   - Link *mutual funds to holdings* via fund_id
   - Link *holdings to stock information* via stock_id
4. Construct a consolidated, feature-rich view for every fund, including all key metadata and top holdings for advanced search and retrieval.

## Intelligent Search Capabilities

- *Natural Language Querying*: Users can search for funds, stocks, or ETFs using natural, conversational, or incomplete queries.
- *Contextual Relevance*: Results are ranked using semantic similarity and enhanced with metadata filtering (sector focus, AUM range, fund house, etc.).
- *Beyond Exact Match*: Captures and understands user intent, even with spelling errors, abbreviations, or partial fund/stock names.

## Example Use Cases

- "Best large cap funds focused on banking"
- "SBI pharma ETFs"
- "Which mutual fund holds most Reliance shares?"
- "Top sectoral mutual funds with AUM > ₹1000 Cr"

## Quick Start

1. Place your JSON datasets in the /data directory.
2. Run the data loading and normalization script to generate the consolidated dataset.
3. Utilize the search API or CLI to interact with the intelligent search engine
