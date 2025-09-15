

class JobMarketplace():

    def __init__(self):
        self.job_listings = []
        self.resumes = []

    def post_job(self, job_title, company, requirements):
        job = {'job_title': job_title, 'company': company, 'requirements': requirements}
        self.job_listings.append(job)
