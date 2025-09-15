

class PageUtil():

    def __init__(self, data, page_size):
        self.data = data
        self.page_size = page_size
        self.total_items = len(data)
        self.total_pages = (((self.total_items + page_size) - 1) // page_size)

    def get_page(self, page_number):
        if ((page_number < 1) or (page_number > self.total_pages)):
            return []
        start_index = ((page_number - 1) * self.page_size)
        end_index = (start_index + self.page_size)
        return self.data[start_index:end_index]
