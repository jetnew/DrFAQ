from openpyxl import load_workbook


class Logger:
    def __init__(self, log_file):
        self.log_file = log_file
        self.workbook = load_workbook(self.log_file)
        self.sheet = self.workbook.active

    def log(self, data):
        assert type(data) == list
        self.sheet.append(data)
        self.workbook.save(filename=self.log_file)
