import scraping_tools.korrekturen_scrape as ortho_data
import scraping_tools.achse_des_guten_api as achse_data

def download_data(sources: list = ['ortho', 'achse', 'zdl', 'gutenberg', 'reddit']):
    if 'ortho' in sources:
        ortho_data.run()
    if 'achse' in sources:
        achse_data.run()