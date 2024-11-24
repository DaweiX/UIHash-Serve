import sys

from collect.main import rename_and_test_apk
from ui2hash.main import apk2data
from ui2hash.database import init_database, insert_data
from ui2hash.compare import search_similar_uis, print_results
    

if __name__ == "__main__":

    # ---------------- step 1 ----------------
    # Input an apk file name, get its UI data
    # ----------------------------------------
    apk = sys.argv[1] if len(sys.argv) > 1 else "1DE2FA20BB138CFE9460689C5458A672AA8C348D3DE362FA4C933F00B0469917.apk"
    if apk == "INITDB":
        init_database()
        exit()

    output_path = rename_and_test_apk(apk)

    # ---------------- step 2 ----------------
    # generate and add UI# results to database
    # ----------------------------------------
    results = apk2data(output_path)
    for result in results:
        insert_data(*result)

    # ---------------- step 3 ----------------
    # compare the app's UIs with existing data
    # ----------------------------------------
    from os.path import basename
    top_k = 5
    apk_sha256 = basename(output_path)
    results = search_similar_uis(apk_sha256, top_k)
    print('-' * 28)
    print(f"Top {top_k} similar UIs detected")
    print('-' * 28)
    print_results(results, apk_sha256)
