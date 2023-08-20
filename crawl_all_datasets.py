import scraping_tools.korrekturen_scrape as ortho_data
import scraping_tools.achse_des_guten_api as achse_data
import scraping_tools.plebbit as reddit_data

def download_data(sources: list = ['ortho', 'achse', 'zdl', 'gutenberg', 'reddit_locales'], path: str = ""):
    if 'ortho' in sources:
        ortho_data.run(path)
    if 'achse' in sources:
        achse_data.run(path)
    if 'reddit_locales' in sources:
        reddit_data.locale_reddits(path)

if __name__ == "__main__":
    download_data(['reddit_locales'], "test/")