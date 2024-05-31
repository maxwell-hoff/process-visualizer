import pandas as pd
import numpy as np
import random
import subprocess
from datetime import datetime, timedelta

class BusinessProcessDataGenerator:
    def __init__(self, num_rows=100000):
        self.num_rows = num_rows
        self.employee_ids = range(1, 501)
        self.client_ids = range(1, 1001)
        self.industries = ['Finance', 'Healthcare', 'Retail', 'Technology', 'Manufacturing']
        self.roles = ['Manager', 'Analyst', 'Developer', 'Consultant', 'Support']
        self.statuses = ['Initiated', 'Reviewed', 'Approved', 'Pending', 'Completed', 'Revised', 'Escalated', 'Resolved', 'Closed', 'Cancelled']
        self.divisions = [f'Division_{i}' for i in range(1, 6)]
        self.departments = {division: [f'{division}_Department_{i}' for i in range(1, 11)] for division in self.divisions}
        self.teams = {department: [f'{department}_Team_{i}' for i in range(1, 21)] for division in self.divisions for department in self.departments[division]}
        self.status_transitions = {
            'Initiated': ['Reviewed', 'Escalated'],
            'Reviewed': ['Approved', 'Revised'],
            'Approved': ['Pending', 'Completed'],
            'Pending': ['Completed', 'Escalated'],
            'Completed': ['Resolved', 'Closed'],
            'Revised': ['Reviewed', 'Approved'],
            'Escalated': ['Reviewed', 'Approved'],
            'Resolved': ['Closed', 'Cancelled'],
            'Closed': [],
            'Cancelled': []
        }

    def random_date(self, start, end):
        return start + timedelta(seconds=random.randint(0, int((end - start).total_seconds())))

    def generate(self):
        data = []
        case_status_history = {}
        employees = []

        for division in self.divisions:
            for department in self.departments[division]:
                for team in self.teams[department]:
                    role = random.choice(self.roles)
                    team_members = random.sample(self.employee_ids, min(10, len(self.employee_ids)))
                    self.employee_ids = [eid for eid in self.employee_ids if eid not in team_members]

                    for employee_id in team_members:
                        employees.append((employee_id, division, department, team, role))

        for _ in range(self.num_rows):
            employee_id, division, department, team, role = random.choice(employees)
            client_id = random.choice(self.client_ids)
            industry = random.choice(self.industries)
            emp_onboard_date = self.random_date(datetime(2000, 1, 1), datetime(2022, 1, 1))
            client_onboard_date = self.random_date(datetime(2000, 1, 1), datetime(2022, 1, 1))
            case_id = random.randint(1, 10000)
            case_created_date = self.random_date(datetime(2000, 1, 1), datetime(2022, 1, 1))
            case_updated_date = case_created_date + timedelta(days=random.randint(1, 1000))
            time_since_created = (case_updated_date - case_created_date).days

            if case_id not in case_status_history:
                case_status_history[case_id] = []

            if not case_status_history[case_id]:
                current_status = 'Initiated'
                time_since_last_modified = time_since_created
            else:
                last_status_record = case_status_history[case_id][-1]
                current_status = self.next_status(last_status_record['current_status'])
                last_modified_date = last_status_record['case_updated_date']
                time_since_last_modified = (case_updated_date - last_modified_date).days

            case_status_history[case_id].append({
                'employee_id': employee_id,
                'division': division,
                'department': department,
                'team': team,
                'role': role,
                'client_id': client_id,
                'industry': industry,
                'emp_onboard_date': emp_onboard_date,
                'client_onboard_date': client_onboard_date,
                'case_id': case_id,
                'case_created_date': case_created_date,
                'case_updated_date': case_updated_date,
                'current_status': current_status,
                'time_since_created': time_since_created,
                'time_since_last_modified': time_since_last_modified
            })

            data.append([
                employee_id, division, department, team, role, client_id, industry,
                emp_onboard_date, client_onboard_date, case_id, case_created_date,
                case_updated_date, current_status, time_since_created, time_since_last_modified
            ])

        columns = [
            'Employee ID', 'Division', 'Department', 'Team', 'Role', 'Client ID', 'Client Industry',
            'Employee Onboarding Date', 'Client Onboarding Date', 'Case ID', 'Case Created Date',
            'Case Updated Date', 'Case Status', 'Time Since Created', 'Time Since Last Modified'
        ]
        df = pd.DataFrame(data, columns=columns)
        return df

    def next_status(self, last_status):
        if last_status in self.status_transitions:
            next_statuses = self.status_transitions[last_status]
            if next_statuses:
                return random.choice(next_statuses)
        return last_status

if __name__ == '__main__':  
    generator = BusinessProcessDataGenerator()
    business_data = generator.generate()
    if 'data/business_process_data.csv' in subprocess.run(['ls', 'data'], capture_output=True).stdout.decode():
        subprocess.run(['rm', 'data/business_process_data.csv'])
    business_data.to_csv('data/business_process_data.csv', index=False)