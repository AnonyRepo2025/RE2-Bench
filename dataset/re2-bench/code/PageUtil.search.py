

class PageUtil():

    def __init__(self, data, page_size):
        self.data = data
        self.page_size = page_size
        self.total_items = len(data)
        self.total_pages = (((self.total_items + page_size) - 1) // page_size)

    def search(self, keyword):
        results = [item for item in self.data if (keyword in str(item))]
        num_results = len(results)
        num_pages = (((num_results + self.page_size) - 1) // self.page_size)
        search_info = {'keyword': keyword, 'total_results': num_results, 'total_pages': num_pages, 'results': results}
        return search_info
