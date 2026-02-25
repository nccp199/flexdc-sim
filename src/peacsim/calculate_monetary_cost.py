

def calculate_monetary_cost(P, R, program_type='NoDR', sim_hour=1, optimization_config=None):
    """
    Calculate the monetary cost of the data center operation based on the power and tracking error.
    """
    HOUR = 3600
    KWH = HOUR * 1000
    piP, piR, piE = 0.1 / KWH, 0.1 / KWH, 0.1 / KWH

    total_duration = HOUR * sim_hour
    if program_type == 'NoDR':       # TODO: Needs refactoring to use the same variable for all
        cost_power = piP * P * total_duration

    elif program_type == 'EDR':
        P_base = optimization_config.p_base
        piI = optimization_config.piI / KWH
        cost_power = (piP * P - piI * (P_base - P)) * total_duration

    elif program_type == 'RSR':
        cost_power = (piP * P - piR * R) * total_duration

    else:
        raise ValueError("Invalid DR type. Choose from ['EDR', 'RSR'].")

    return cost_power


def calculate_total_monetary_cost(P, R, program_type='RSR', sim_hour=1, avg_tracking_error=None):
    """
    Calculate the total monetary cost of the data center operation based on the power and average tracking error.
    """

    HOUR = 3600
    KWH = HOUR * 1000
    piP, piR, piE = 0.1 / KWH, 0.1 / KWH, 0.1 / KWH

    total_duration = HOUR * sim_hour
    if program_type == 'RSR':
        cost_power = (piP * P - piR * R) * total_duration
        cost_tracking_error = piE * avg_tracking_error * total_duration

    else:
        raise ValueError("Invalid DR type. Choose from ['EDR', 'RSR'].")

    return cost_power + cost_tracking_error
