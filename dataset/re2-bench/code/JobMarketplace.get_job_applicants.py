

class JobMarketplace():

    def __init__(self):
        self.job_listings = []
        self.resumes = []

    def get_job_applicants(self, job):
        applicants = []
        for resume in self.resumes:
            if self.matches_requirements(resume, job['requirements']):
                applicants.append(resume)
        return applicants

    @staticmethod
    def matches_requirements(resume, requirements):
        for skill in resume['skills']:
            if (skill not in requirements):
                return False
        return True
