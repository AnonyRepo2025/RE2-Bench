import json


def create_report_json_file(models_task_results, paths):
    metadata_path = paths.metadata_path
    with open(metadata_path, 'w') as f:
        json.dump(models_task_results, f, indent=2)


def create_detail_csv_file(results, paths):
    output_path = paths.output_path
    import csv
    with open(output_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=results[0].keys())
        writer.writeheader()
        writer.writerows(results)


class ValidationOutputPaths:
    def __init__(self, csv_detailed_path, json_report_path):
        self.output_path = csv_detailed_path
        self.metadata_path = json_report_path
