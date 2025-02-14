#!/usr/bin/env python3
"""
Script to test and compare responses from the FTS-based and embeddings-based searches.
For each query, the script prints an ideal expected extract, displays the search results,
and compares whether the ideal expected extract is included in the results.
"""

import time

from prettytable import PrettyTable
from pysqlite3 import dbapi2 as sqlite3
from sqlite_vec import load

from rag import get_model, get_reranker, search, search_embeddings, search_rerank

# Define expected ideal extracts for each query.
expected_responses = {
    "seguridad y tecnología": "Cifrado de extremo a extremo: Toda la comunicación dentro de la app está cifrada",
    "educación financiera": "Artículos y guías: Una extensa biblioteca de contenido educativo",
    "expansión internacional": "Expansión Nacional e Internacional",
    "beneficios de usar una tarjeta": "Acceso a promociones exclusivas: Ofertas especiales en comercios afiliados",
    "préstamos personales": "Tasa de interés: Competitiva, basada en el perfil crediticio del usuario, con la posibilidad de reducir la tasa mediante pagos puntuales.",
    "configuración": "La sección de Configuración permite a los usuarios personalizar la app de acuerdo con sus preferencias y necesidades de seguridad.",
    "tarjeta de crédito": "es un producto clave para aquellos que buscan construir o mejorar su historial crediticio.",
    "tarjeta de débito": "ofrece una experiencia bancaria completa, segura y sin complicaciones.",
    "autenticación biométrica": "Los usuarios pueden acceder a su cuenta mediante huella digital o reconocimiento facial",
}

LIMIT = 10


def run_test(query: str):
    db = sqlite3.connect("./db.sqlite")
    db.enable_load_extension(True)
    load(db)
    db.enable_load_extension(False)
    print("=" * 40)
    print(f"Query: {query}\n")

    expected = expected_responses.get(query, "No expected response defined.")
    print("Ideal Expected Extract:")
    print(expected)
    start_fts = time.perf_counter()
    results_fts = search(db, query, LIMIT)
    end_fts = time.perf_counter()
    fts_time = end_fts - start_fts

    _ = get_model()  # load model to avoid timing it
    start_emb = time.perf_counter()
    results_emb = search_embeddings(db, query, LIMIT)
    end_emb = time.perf_counter()
    emb_time = end_emb - start_emb

    _ = get_reranker()  # load reranker to avoid timing it
    start_rer = time.perf_counter()
    tmp_results_emb = search_embeddings(db, query, LIMIT * 2)
    results_rer = search_rerank(db, tmp_results_emb, query)
    end_rer = time.perf_counter()
    rer_time = end_rer - start_rer

    def check_results(results):
        for i, row in enumerate(results):
            if expected in row[2]:
                return i
        return LIMIT * 2  # not found in "limit" results, penalize with * 2

    fts_match = check_results(results_fts)
    emb_match = check_results(results_emb)
    rer_match = check_results(results_rer)
    summary = f"FTS found on: {fts_match}\nEmbeddings found on: {emb_match}\nReranking found on: {rer_match}"
    print("\nTest result:")
    print(summary)
    print("=" * 40, "\n")
    return [fts_match, emb_match, rer_match, fts_time, emb_time, rer_time]


def main() -> None:
    fts_total = 0
    emb_total = 0
    rer_total = 0
    fts_time_total = 0
    emb_time_total = 0
    rer_time_total = 0
    table = PrettyTable()
    table.field_names = [
        "Query",
        "FTS",
        "Embeddings",
        "Reranker",
        "FTS Time (s)",
        "Embeddings Time (s)",
        "Reranker Time (s)",
    ]
    for query in expected_responses.keys():
        fts_i, emb_i, rer_i, fts_time, emb_time, rer_time = run_test(query)
        fts_total += fts_i
        emb_total += emb_i
        rer_total += rer_i
        fts_time_total += fts_time
        emb_time_total += emb_time
        rer_time_total += rer_time
        table.add_row(
            [
                query,
                fts_i,
                emb_i,
                rer_i,
                f"{fts_time:.6f}",
                f"{emb_time:.6f}",
                f"{rer_time:.6f}",
            ]
        )

    # print summary
    print("Summary:")
    print(
        f"Note. Search limit was {LIMIT}, if you see {LIMIT * 2} it means the expected extract was not found."
    )
    print(table)
    print(
        f"FTS total: {fts_total}, Embeddings total: {emb_total}, Reranker total: {rer_total}. LESS IS MORE"
    )
    print(
        f"Total FTS Search Time: {fts_time_total:.6f} seconds, Total Embeddings Search Time: {emb_time_total:.6f} seconds, Total Reranker Time: {rer_time_total:.6f} seconds"
    )


if __name__ == "__main__":
    main()
