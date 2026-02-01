from app.bootstrap import bootstrap
from app.run_paper_live import run

def main():
    cfg, con, td, logger = bootstrap()
    run(cfg, con, td, logger, model_path="model/artifacts/gb_model.joblib")

if __name__ == "__main__":
    main()
