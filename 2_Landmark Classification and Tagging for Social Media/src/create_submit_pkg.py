import datetime
import glob
import tarfile

import nbformat
from nbconvert import HTMLExporter


def create_submit_pkg():

    # Source files
    src_files = glob.glob("src/*.py")

    # Notebooks
    notebooks = glob.glob("*.ipynb")

    # Generate HTML files from the notebooks (using nbconvert Python API)
    for nb in notebooks:
        print(f"executing: convert {nb} -> HTML")
        with open(nb, encoding="utf-8") as f:
            nb_node = nbformat.read(f, as_version=4)
        (body, _) = HTMLExporter().from_notebook_node(nb_node)
        out_path = nb.replace(".ipynb", ".html")
        with open(out_path, "w", encoding="utf-8") as f:
            f.write(body)

    html_files = glob.glob("*.htm*")

    now = datetime.datetime.today().isoformat(timespec="minutes").replace(":", "h")+"m"
    outfile = f"submission_{now}.tar.gz"
    print(f"Adding files to {outfile}")
    with tarfile.open(outfile, "w:gz") as tar:
        for name in (src_files + notebooks + html_files):
            print(name)
            tar.add(name)

    print("")
    msg = f"Done. Please submit the file {outfile}"
    print("-" * len(msg))
    print(msg)
    print("-" * len(msg))


if __name__ == "__main__":
    create_submit_pkg()