import configparser
import os


def write_workload(applications_file, applications_dict, workload_name):
    """Writes the workload config file to '../../../configs/workload/[workload_name].ini'."""
    output_file = f"../../../configs/workload/{workload_name}.ini"

    apps_parser = configparser.ConfigParser()
    apps_parser.read(applications_file)

    output_parser = configparser.ConfigParser()

    for app, job_size in applications_dict.items():
        if apps_parser.has_section(app):
            output_parser[app] = {key: apps_parser.get(app, key) for key in apps_parser.options(app)}
            output_parser[app]['job_size'] = str(job_size)  # Add job_size from dictionary

    with open(output_file, "w") as configfile:
        output_parser.write(configfile)


if __name__ == "__main__":
    app_file = "../../../configs/job/scc-applications.ini"

    # W8
    application_to_size_dict = {"bt.C.16": 1, "is.D.32": 3, "bt.C.36": 2, "cg.C.4": 1, "mg.D.8": 1, "cg.D.16": 1, "cg.D.128": 6, "cg.D.32": 3}
    write_workload(app_file , application_to_size_dict, "W8")

    # W9
    application_to_size_dict = {"mg.D.16": 1, "sp.C.16": 1, "bt.D.49": 2, "ep.D.64": 3, "sp.D.100": 4, "ep.D.28": 1,
                         "bt.D.25": 1, "mg.D.32": 2}
    write_workload(app_file , application_to_size_dict, "W9")

    # W11
    application_to_size_dict = {"sp.C.16": 1, "is.D.32": 3, "bt.D.49": 2, "ep.D.64": 3, "lu.D.224": 8, "sp.D.16": 1,
                         "cg.D.16": 1, "cg.D.32": 3, "mg.D.32": 2}
    write_workload(app_file, application_to_size_dict, "W11")

    # W12
    application_to_size_dict = {"is.D.32": 3, "bt.D.49": 2, "mg.D.8": 1, "is.D.4": 1, "ep.D.100": 4, "is.D.64": 4,
                         "lu.D.112": 4, "mg.D.32": 2}
    write_workload(app_file, application_to_size_dict, "W12")


