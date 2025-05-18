import sqlite3, pathlib

DB = pathlib.Path(__file__).parent / "tokens.db"
with sqlite3.connect(DB) as con:
    con.execute("""CREATE TABLE IF NOT EXISTS vault
                   (epoch INT, donor TEXT, tokens REAL)""")

def mint(epoch: int, donor: str, tokens: float) -> None:
    with sqlite3.connect(DB) as con, con:
        con.execute("INSERT INTO vault VALUES (?,?,?)",
                    (epoch, donor, tokens))

def load(epoch: int, expiry: int):
    with sqlite3.connect(DB) as con:
        return con.execute("SELECT donor,tokens FROM vault WHERE epoch>=?",
                           (epoch - expiry,)).fetchall()

def expire(epoch: int, expiry: int) -> None:
    with sqlite3.connect(DB) as con, con:
        con.execute("DELETE FROM vault WHERE epoch<?", (epoch - expiry,))
