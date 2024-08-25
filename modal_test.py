import os
import modal

app = modal.App()


@app.function(secrets=[modal.Secret.from_dotenv()])
def some_other_function():
    print(os.environ["HF_TOKEN"])
