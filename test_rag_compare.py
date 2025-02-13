#!/usr/bin/env python3
"""
Script to test and compare responses from the FTS-based and embeddings-based searches.
For each query, the script prints an ideal expected extract, displays the search results,
and compares whether the ideal expected extract is included in the results.
"""

from prettytable import PrettyTable
from pysqlite3 import dbapi2 as sqlite3
from sqlite_vec import load

from rag import search, search_embeddings

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
    print("\nFTS Results:")
    results_fts = search(db, query, LIMIT)
    for row in results_fts:
        print(f"ID: {row[0]}, Rank: {row[3]}")
        print(row[2])
        print("-" * 20)

    print("\nEmbeddings Results:")
    results_emb = search_embeddings(db, query, LIMIT)
    for row in results_emb:
        print(f"ID: {row[0]}, Rank: {row[3]}")
        print(row[2])
        print("-" * 20)

    def check_results(results):
        for i, row in enumerate(results):
            if expected in row[2]:
                return i
        return LIMIT * 2  # not found in "limit" results, penalize with * 2

    fts_match = check_results(results_fts)
    emb_match = check_results(results_emb)
    summary = f"FTS found on: {fts_match}, Embeddings found on: {emb_match}. "
    if fts_match < emb_match:
        summary += "Recommended approach: FTS."
    elif emb_match < fts_match:
        summary += "Recommended approach: Embeddings."
    elif fts_match == emb_match and fts_match < LIMIT:
        summary += "Both methods work. Consider FTS due to its simplicity."
    else:
        summary += "Neither method found the expected extract."
    print("\nTest result:")
    print(summary)
    print("=" * 40, "\n")
    return [fts_match, emb_match]


def main() -> None:
    fts_total = 0
    emb_total = 0
    table = PrettyTable()
    table.field_names = ["Query", "FTS", "Embeddings"]
    for query in expected_responses.keys():
        fts_i, emb_i = run_test(query)
        fts_total += fts_i
        emb_total += emb_i
        table.add_row([query, fts_i, emb_i])

    # print summary
    print("Summary:")
    print(
        f"Note. Search limit was {LIMIT}, if you see {LIMIT * 2} it means the expected extract was not found."
    )
    print(table)
    print(f"FTS total: {fts_total}, Embeddings total: {emb_total}. LESS IS MORE")


if __name__ == "__main__":
    main()
