from pathlib import Path


def parse_libero_report(out_dir):
    """Reads and parses an FPGA synthesis report into a structured dictionary."""

    out_path = Path(out_dir)
    report_path = out_path / 'hls_output/reports/summary.results.rpt'
    if not report_path.exists():
        print(f'Libero report file {str(report_path)} not found.')
        return {}

    with open(report_path) as file:
        report_lines = file.readlines()

    return {
        'Simulation Result': _parse_sim_data(_extract_section(report_lines, '====== 1. Simulation Result')),
        'Timing Result': _parse_timing_data(_extract_section(report_lines, '====== 2. Timing Result')),
        'Resource Usage': parse_utilization_data(_extract_section(report_lines, '====== 3. Resource Usage')),
    }


def _extract_section(lines, section_header):
    """Extracts table data from a given section in the report."""

    section_data = []
    in_section = False

    for line in lines:
        if section_header in line:
            in_section = True
            continue

        if in_section:
            if line.startswith('======'):  # Start of next section
                break
            section_data.append(line.strip())

    return section_data


def _parse_sim_data(data):
    """Parses 'Simulation Result' section."""
    if len(data) == 0:
        return {'Error': 'Data missing for this section'}

    sim_dict = {}

    for line in data:
        if line.startswith('N/A. Please run'):
            return {'Error': line}
        elif line.startswith('+') or line.startswith('| Top-Level Name'):
            continue  # Ignore table borders
        elif '|' in line:
            columns = [col.strip() for col in line.split('|')[2:-1]]
            sim_dict.update(
                {
                    'Number of calls': columns[0],
                    'Simulation time (cycles)': columns[1],
                    'Call Latency (min/max/avg)': columns[2],
                    'Call II (min/max/avg)': columns[3],
                }
            )
        elif 'SW/HW co-simulation' in line:
            sim_dict['Status'] = line.split(':')[1].strip()

    return sim_dict


def _parse_timing_data(data):
    """Parses 'Timing Result' section."""
    if len(data) == 0:
        return {'Error': 'Data missing for this section'}

    timing_dict = {}

    for line in data:
        if line.startswith('N/A. Please run'):
            return {'Error': line}
        elif line.startswith('+') or line.startswith('| Clock Domain'):
            continue  # Ignore table borders
        elif '|' in line:
            columns = [col.strip() for col in line.split('|')[2:-1]]
            timing_dict.update(
                {
                    'Target Period': columns[0],
                    'Target Fmax': columns[1],
                    'Worst Slack': columns[2],
                    'Period': columns[3],
                    'Fmax': columns[4],
                }
            )

    return timing_dict


def parse_utilization_data(data):
    """Parses 'Resource Usage' section."""
    if len(data) == 0:
        return {'Error': 'Data missing for this section'}

    util_dict = {}

    for line in data:
        if line.startswith('N/A. Please run'):
            return {'Error': line}
        elif line.startswith('+') or line.startswith('| Resource Type'):
            continue  # Ignore table borders
        elif '|' in line:
            columns = [col.strip() for col in line.split('|')[1:-1]]
            util_dict[columns[0]] = {
                'Used': columns[1],
                'Total': columns[2],
                'Percentage': columns[3],
            }

    return util_dict
