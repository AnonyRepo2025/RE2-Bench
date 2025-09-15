import inspect
import json
import os
from datetime import datetime

def custom_serializer(obj):
    if isinstance(obj, datetime):
        return obj.isoformat()
    return str(obj)


def recursive_object_seralizer(obj, visited):
    seralized_dict = {}
    keys = list(obj.__dict__)
    for k in keys:
        if id(obj.__dict__[k]) in visited:
            seralized_dict[k] = "<RECURSIVE {}>".format(obj.__dict__[k])
            continue
        if isinstance(obj.__dict__[k], (float, int, str, bool, type(None))):
            seralized_dict[k] = obj.__dict__[k]
        elif isinstance(obj.__dict__[k], tuple):
            ## handle tuple
            seralized_dict[k] = obj.__dict__[k]
        elif isinstance(obj.__dict__[k], set):
            ## handle set
            seralized_dict[k] = obj.__dict__[k]
        elif isinstance(obj.__dict__[k], list):
            ## handle list
            seralized_dict[k] = obj.__dict__[k]
        elif hasattr(obj.__dict__[k], '__dict__'):
            ## handle object
            visited.append(id(obj.__dict__[k]))
            seralized_dict[k] = obj.__dict__[k]
        elif isinstance(obj.__dict__[k], dict):
            visited.append(id(obj.__dict__[k]))
            seralized_dict[k] = obj.__dict__[k]
        elif callable(obj.__dict__[k]):
            ## handle function
            if hasattr(obj.__dict__[k], '__name__'):
                seralized_dict[k] = "<function {}>".format(obj.__dict__[k].__name__)
        else:
            seralized_dict[k] = str(obj.__dict__[k])
    return seralized_dict

def inspect_code(func):
   def wrapper(*args, **kwargs):
       visited = []
       json_base = "/home/changshu/ClassEval/data/benchmark_solution_code/input-output/"
       if not os.path.exists(json_base):
           os.mkdir(json_base)
       jsonl_path = json_base + "/JobMarketplace.jsonl"
       para_dict = {"name": func.__name__}
       args_names = inspect.getfullargspec(func).args
       if len(args) > 0 and hasattr(args[0], '__dict__') and args_names[0] == 'self':
           ## 'self'
           self_args = args[0]
           para_dict['self'] = recursive_object_seralizer(self_args, [id(self_args)])
       else:
           para_dict['self'] = {}
       if len(args) > 0 :
           if args_names[0] == 'self':
               other_args = {}
               for m,n in zip(args_names[1:], args[1:]):
                   other_args[m] = n
           else:
               other_args = {}
               for m,n in zip(args_names, args):
                   other_args[m] = n
           
           para_dict['args'] = other_args
       else:
           para_dict['args'] = {}
       if kwargs:
           para_dict['kwargs'] = kwargs
       else:
           para_dict['kwargs'] = {}
          
       result = func(*args, **kwargs)
       para_dict["return"] = result
       with open(jsonl_path, 'a') as f:
           f.write(json.dumps(para_dict, default=custom_serializer) + "\n")
       return result
   return wrapper


'''
# This is a class that provides functionalities to publish positions, remove positions, submit resumes, withdraw resumes, search for positions, and obtain candidate information.

class JobMarketplace:
    def __init__(self):
        self.job_listings = []
        self.resumes = []

    def post_job(self, job_title, company, requirements):
        """
        This function is used to publish positions,and add the position information to the job_listings list.
        :param job_title: The title of the position,str.
        :param company: The company of the position,str.
        :param requirements: The requirements of the position,list.
        :return: None
        >>> jobMarketplace = JobMarketplace()
        >>> jobMarketplace.post_job("Software Engineer", "ABC Company", ['requirement1', 'requirement2'])
        >>> jobMarketplace.job_listings
        [{'job_title': 'Software Engineer', 'company': 'ABC Company', 'requirements': ['requirement1', 'requirement2']}]

        """

    def remove_job(self, job):
        """
        This function is used to remove positions,and remove the position information from the job_listings list.
        :param job: The position information to be removed,dict.
        :return: None
        >>> jobMarketplace = JobMarketplace()
        >>> jobMarketplace.job_listings = [{"job_title": "Software Engineer", "company": "ABC Company", "requirements": ['requirement1', 'requirement2']}]
        >>> jobMarketplace.remove_job(jobMarketplace.job_listings[0])
        >>> jobMarketplace.job_listings
        []

        """

    def submit_resume(self, name, skills, experience):
        """
        This function is used to submit resumes,and add the resume information to the resumes list.
        :param name: The name of the resume,str.
        :param skills: The skills of the resume,list.
        :param experience: The experience of the resume,str.
        :return: None
        >>> jobMarketplace = JobMarketplace()
        >>> jobMarketplace.submit_resume("Tom", ['skill1', 'skill2'], "experience")
        >>> jobMarketplace.resumes
        [{'name': 'Tom', 'skills': ['skill1', 'skill2'], 'experience': 'experience'}]

        """

    def withdraw_resume(self, resume):
        """
        This function is used to withdraw resumes,and remove the resume information from the resumes list.
        :param resume: The resume information to be removed,dict.
        :return: None
        >>> jobMarketplace = JobMarketplace()
        >>> jobMarketplace.resumes = [{"name": "Tom", "skills": ['skill1', 'skill2'], "experience": "experience"}]
        >>> jobMarketplace.withdraw_resume(jobMarketplace.resumes[0])
        >>> jobMarketplace.resumes
        []

        """

    def search_jobs(self, criteria):
        """
        This function is used to search for positions,and return the position information that meets the requirements.
        :param criteria: The requirements of the position,str.
        :return: The position information that meets the requirements,list.
        >>> jobMarketplace = JobMarketplace()
        >>> jobMarketplace.job_listings = [{"job_title": "Software Engineer", "company": "ABC Company", "requirements": ['skill1', 'skill2']}]
        >>> jobMarketplace.search_jobs("skill1")
        [{'job_title': 'Software Engineer', 'company': 'ABC Company', 'requirements': ['skill1', 'skill2']}]

        """

    def get_job_applicants(self, job):
        """
        This function is used to obtain candidate information,and return the candidate information that meets the requirements by calling the matches_requirements function.
        :param job: The position information,dict.
        :return: The candidate information that meets the requirements,list.
        >>> jobMarketplace = JobMarketplace()
        >>> jobMarketplace.resumes = [{"name": "Tom", "skills": ['skill1', 'skill2'], "experience": "experience"}]
        >>> jobMarketplace.job_listings = [{"job_title": "Software Engineer", "company": "ABC Company", "requirements": ['skill1', 'skill2']}]
        >>> jobMarketplace.get_job_applicants(jobMarketplace.job_listings[0])
        [{'name': 'Tom', 'skills': ['skill1', 'skill2'], 'experience': 'experience'}]

        """



'''

class JobMarketplace:
    def __init__(self):
        self.job_listings = []
        self.resumes = []

    @inspect_code
    def post_job(self, job_title, company, requirements):
        # requirements = ['requirement1', 'requirement2']
        job = {"job_title": job_title, "company": company, "requirements": requirements}
        self.job_listings.append(job)

    @inspect_code
    def remove_job(self, job):
        self.job_listings.remove(job)

    @inspect_code
    def submit_resume(self, name, skills, experience):
        resume = {"name": name, "skills": skills, "experience": experience}
        self.resumes.append(resume)

    @inspect_code
    def withdraw_resume(self, resume):
        self.resumes.remove(resume)

    @inspect_code
    def search_jobs(self, criteria):
        matching_jobs = []
        for job_listing in self.job_listings:
            if criteria.lower() in job_listing["job_title"].lower() or criteria.lower() in [r.lower() for r in job_listing["requirements"]]:
                matching_jobs.append(job_listing)
        return matching_jobs

    @inspect_code
    def get_job_applicants(self, job):
        applicants = []
        for resume in self.resumes:
            if self.matches_requirements(resume, job["requirements"]):
                applicants.append(resume)
        return applicants

    @staticmethod
    @inspect_code
    def matches_requirements(resume, requirements):
        for skill in resume["skills"]:
            if skill not in requirements:
                return False
        return True

import unittest
class JobMarketplaceTestPostJob(unittest.TestCase):
    def test_post_job(self):
        jobMarketplace = JobMarketplace()
        jobMarketplace.post_job("Software Engineer", "ABC Company", ['requirement1', 'requirement2'])
        self.assertEqual(jobMarketplace.job_listings, [{'job_title': 'Software Engineer', 'company': 'ABC Company', 'requirements': ['requirement1', 'requirement2']}])

    def test_post_job_2(self):
        jobMarketplace = JobMarketplace()
        jobMarketplace.post_job("Mechanical Engineer", "XYZ Company", ['requirement3', 'requirement4'])
        self.assertEqual(jobMarketplace.job_listings, [{'job_title': 'Mechanical Engineer', 'company': 'XYZ Company', 'requirements': ['requirement3', 'requirement4']}])

    def test_post_job_3(self):
        jobMarketplace = JobMarketplace()
        jobMarketplace.post_job("Software Engineer", "ABC Company", ['requirement1', 'requirement2'])
        jobMarketplace.post_job("Mechanical Engineer", "XYZ Company", ['requirement3', 'requirement4'])
        self.assertEqual(jobMarketplace.job_listings, [{'job_title': 'Software Engineer', 'company': 'ABC Company', 'requirements': ['requirement1', 'requirement2']}, {'job_title': 'Mechanical Engineer', 'company': 'XYZ Company', 'requirements': ['requirement3', 'requirement4']}])

    def test_post_job_4(self):
        jobMarketplace = JobMarketplace()
        jobMarketplace.post_job("Software Engineer", "ABC Company", ['requirement1', 'requirement2'])
        jobMarketplace.post_job("Mechanical Engineer", "XYZ Company", ['requirement3', 'requirement4'])
        jobMarketplace.post_job("Software Engineer", "ABC Company", ['requirement1', 'requirement2'])
        self.assertEqual(jobMarketplace.job_listings, [{'job_title': 'Software Engineer', 'company': 'ABC Company', 'requirements': ['requirement1', 'requirement2']}, {'job_title': 'Mechanical Engineer', 'company': 'XYZ Company', 'requirements': ['requirement3', 'requirement4']}, {'job_title': 'Software Engineer', 'company': 'ABC Company', 'requirements': ['requirement1', 'requirement2']}])

    def test_post_job_5(self):
        jobMarketplace = JobMarketplace()
        jobMarketplace.post_job("Software Engineer", "ABC Company", ['requirement1', 'requirement2'])
        jobMarketplace.post_job("Mechanical Engineer", "XYZ Company", ['requirement3', 'requirement4'])
        jobMarketplace.post_job("Software Engineer", "ABC Company", ['requirement1', 'requirement2'])
        jobMarketplace.post_job("Mechanical Engineer", "XYZ Company", ['requirement3', 'requirement4'])
        self.assertEqual(jobMarketplace.job_listings, [{'job_title': 'Software Engineer', 'company': 'ABC Company', 'requirements': ['requirement1', 'requirement2']}, {'job_title': 'Mechanical Engineer', 'company': 'XYZ Company', 'requirements': ['requirement3', 'requirement4']}, {'job_title': 'Software Engineer', 'company': 'ABC Company', 'requirements': ['requirement1', 'requirement2']}, {'job_title': 'Mechanical Engineer', 'company': 'XYZ Company', 'requirements': ['requirement3', 'requirement4']}])

class JobMarketplaceTestRemoveJob(unittest.TestCase):
    def test_remove_job(self):
        jobMarketplace = JobMarketplace()
        jobMarketplace.job_listings = [{"job_title": "Software Engineer", "company": "ABC Company", "requirements": ['requirement1', 'requirement2']}]
        jobMarketplace.remove_job(jobMarketplace.job_listings[0])
        self.assertEqual(jobMarketplace.job_listings, [])

    def test_remove_job_2(self):
        jobMarketplace = JobMarketplace()
        jobMarketplace.job_listings = [{"job_title": "Software Engineer", "company": "ABC Company", "requirements": ['requirement1', 'requirement2']}, {"job_title": "Mechanical Engineer", "company": "XYZ Company", "requirements": ['requirement3', 'requirement4']}]
        jobMarketplace.remove_job(jobMarketplace.job_listings[0])
        self.assertEqual(jobMarketplace.job_listings, [{'job_title': 'Mechanical Engineer', 'company': 'XYZ Company', 'requirements': ['requirement3', 'requirement4']}])

    def test_remove_job_3(self):
        jobMarketplace = JobMarketplace()
        jobMarketplace.job_listings = [{"job_title": "Software Engineer", "company": "ABC Company", "requirements": ['requirement1', 'requirement2']}, {"job_title": "Mechanical Engineer", "company": "XYZ Company", "requirements": ['requirement3', 'requirement4']}]
        jobMarketplace.remove_job(jobMarketplace.job_listings[0])
        jobMarketplace.remove_job(jobMarketplace.job_listings[0])
        self.assertEqual(jobMarketplace.job_listings, [])

    def test_remove_job_4(self):
        jobMarketplace = JobMarketplace()
        jobMarketplace.job_listings = [{"job_title": "Software Engineer", "company": "ABC Company", "requirements": ['requirement1', 'requirement2']}, {"job_title": "Mechanical Engineer", "company": "XYZ Company", "requirements": ['requirement3', 'requirement4']}, {"job_title": "Software Engineer", "company": "ABC Company", "requirements": ['requirement1', 'requirement2']}]
        jobMarketplace.remove_job(jobMarketplace.job_listings[0])
        jobMarketplace.remove_job(jobMarketplace.job_listings[0])
        self.assertEqual(jobMarketplace.job_listings, [{'job_title': 'Software Engineer', 'company': 'ABC Company', 'requirements': ['requirement1', 'requirement2']}])

    def test_remove_job_5(self):
        jobMarketplace = JobMarketplace()
        jobMarketplace.job_listings = [{"job_title": "Software Engineer", "company": "ABC Company",
                                       "requirements": ['requirement1', 'requirement2']},
                                      {"job_title": "Mechanical Engineer", "company": "XYZ Company",
                                       "requirements": ['requirement3', 'requirement4']},
                                      {"job_title": "Software Engineer", "company": "ABC Company",
                                       "requirements": ['requirement1', 'requirement2']}]
        jobMarketplace.remove_job(jobMarketplace.job_listings[0])
        self.assertEqual(jobMarketplace.job_listings, [{'job_title': 'Mechanical Engineer', 'company': 'XYZ Company', 'requirements': ['requirement3', 'requirement4']}, {'job_title': 'Software Engineer', 'company': 'ABC Company', 'requirements': ['requirement1', 'requirement2']}])

class JobMarketplaceTestSubmitResume(unittest.TestCase):
    def test_submit_resume(self):
        jobMarketplace = JobMarketplace()
        jobMarketplace.submit_resume("Tom", ['skill1', 'skill2'], "experience")
        self.assertEqual(jobMarketplace.resumes, [{'name': 'Tom', 'skills': ['skill1', 'skill2'], 'experience': 'experience'}])

    def test_submit_resume_2(self):
        jobMarketplace = JobMarketplace()
        jobMarketplace.submit_resume("Tom", ['skill1', 'skill2'], "experience")
        jobMarketplace.submit_resume("John", ['skill3', 'skill4'], "experience")
        self.assertEqual(jobMarketplace.resumes, [{'name': 'Tom', 'skills': ['skill1', 'skill2'], 'experience': 'experience'}, {'name': 'John', 'skills': ['skill3', 'skill4'], 'experience': 'experience'}])

    def test_submit_resume_3(self):
        jobMarketplace = JobMarketplace()
        jobMarketplace.submit_resume("Tom", ['skill1', 'skill2'], "experience")
        jobMarketplace.submit_resume("John", ['skill3', 'skill4'], "experience")
        jobMarketplace.submit_resume("Tom", ['skill1', 'skill2'], "experience")
        self.assertEqual(jobMarketplace.resumes, [{'name': 'Tom', 'skills': ['skill1', 'skill2'], 'experience': 'experience'}, {'name': 'John', 'skills': ['skill3', 'skill4'], 'experience': 'experience'}, {'name': 'Tom', 'skills': ['skill1', 'skill2'], 'experience': 'experience'}])

    def test_submit_resume_4(self):
        jobMarketplace = JobMarketplace()
        jobMarketplace.submit_resume("Tom", ['skill1', 'skill2'], "experience")
        jobMarketplace.submit_resume("John", ['skill3', 'skill4'], "experience")
        jobMarketplace.submit_resume("Tom", ['skill1', 'skill2'], "experience")
        jobMarketplace.submit_resume("John", ['skill3', 'skill4'], "experience")
        self.assertEqual(jobMarketplace.resumes, [{'name': 'Tom', 'skills': ['skill1', 'skill2'], 'experience': 'experience'}, {'name': 'John', 'skills': ['skill3', 'skill4'], 'experience': 'experience'}, {'name': 'Tom', 'skills': ['skill1', 'skill2'], 'experience': 'experience'}, {'name': 'John', 'skills': ['skill3', 'skill4'], 'experience': 'experience'}])

    def test_submit_resume_5(self):
        jobMarketplace = JobMarketplace()
        jobMarketplace.submit_resume("Tom", ['skill1', 'skill2'], "experience")
        jobMarketplace.submit_resume("John", ['skill3', 'skill4'], "experience")
        jobMarketplace.submit_resume("Tom", ['skill1', 'skill2'], "experience")
        jobMarketplace.submit_resume("John", ['skill3', 'skill4'], "experience")
        jobMarketplace.submit_resume("Tom", ['skill1', 'skill2'], "experience")
        self.assertEqual(jobMarketplace.resumes, [{'name': 'Tom', 'skills': ['skill1', 'skill2'], 'experience': 'experience'}, {'name': 'John', 'skills': ['skill3', 'skill4'], 'experience': 'experience'}, {'name': 'Tom', 'skills': ['skill1', 'skill2'], 'experience': 'experience'}, {'name': 'John', 'skills': ['skill3', 'skill4'], 'experience': 'experience'}, {'name': 'Tom', 'skills': ['skill1', 'skill2'], 'experience': 'experience'}])


class JobMarketplaceTestWithdrawResume(unittest.TestCase):
    def test_withdraw_resume(self):
        jobMarketplace = JobMarketplace()
        jobMarketplace.resumes = [{"name": "Tom", "skills": ['skill1', 'skill2'], "experience": "experience"}]
        jobMarketplace.withdraw_resume(jobMarketplace.resumes[0])
        self.assertEqual(jobMarketplace.resumes, [])

    def test_withdraw_resume_2(self):
        jobMarketplace = JobMarketplace()
        jobMarketplace.resumes = [{"name": "Tom", "skills": ['skill1', 'skill2'], "experience": "experience"}, {"name": "John", "skills": ['skill3', 'skill4'], "experience": "experience"}]
        jobMarketplace.withdraw_resume(jobMarketplace.resumes[0])
        self.assertEqual(jobMarketplace.resumes, [{'name': 'John', 'skills': ['skill3', 'skill4'], 'experience': 'experience'}])

    def test_withdraw_resume_3(self):
        jobMarketplace = JobMarketplace()
        jobMarketplace.resumes = [{"name": "Tom", "skills": ['skill1', 'skill2'], "experience": "experience"}, {"name": "John", "skills": ['skill3', 'skill4'], "experience": "experience"}]
        jobMarketplace.withdraw_resume(jobMarketplace.resumes[0])
        jobMarketplace.withdraw_resume(jobMarketplace.resumes[0])
        self.assertEqual(jobMarketplace.resumes, [])
    
    def test_withdraw_resume_4(self):
        jobMarketplace = JobMarketplace()
        jobMarketplace.resumes = [{"name": "Amy", "skills": ['skill3', 'skill2'], "experience": "experience"}, {"name": "John", "skills": ['skill3', 'skill4'], "experience": "experience"}]
        jobMarketplace.withdraw_resume(jobMarketplace.resumes[0])
        jobMarketplace.withdraw_resume(jobMarketplace.resumes[0])
        self.assertEqual(jobMarketplace.resumes, [])

    def test_withdraw_resume_5(self):
        jobMarketplace = JobMarketplace()
        jobMarketplace.resumes = [{"name": "Amy", "skills": ['skill1', 'skill2'], "experience": "experience"}, {"name": "John", "skills": ['skill3', 'skill4'], "experience": "experience"}]
        jobMarketplace.withdraw_resume(jobMarketplace.resumes[0])
        self.assertEqual(jobMarketplace.resumes, [{'experience': 'experience', 'name': 'John', 'skills': ['skill3', 'skill4']}])

class JobMarketplaceTestSearchJobs(unittest.TestCase):
    def test_search_jobs(self):
        jobMarketplace = JobMarketplace()
        jobMarketplace.job_listings = [{"job_title": "Software Engineer", "company": "ABC Company", "requirements": ['skill1', 'skill2']}]
        self.assertEqual(jobMarketplace.search_jobs("skill1"), [{'job_title': 'Software Engineer', 'company': 'ABC Company', 'requirements': ['skill1', 'skill2']}])

    def test_search_jobs_2(self):
        jobMarketplace = JobMarketplace()
        jobMarketplace.job_listings = [{"job_title": "Software Engineer", "company": "ABC Company", "requirements": ['skill1', 'skill2']}, {"job_title": "Software Engineer", "company": "ABC Company", "requirements": ['skill3', 'skill4']}]
        self.assertEqual(jobMarketplace.search_jobs("skill1"), [{'job_title': 'Software Engineer', 'company': 'ABC Company', 'requirements': ['skill1', 'skill2']}])

    def test_search_jobs_3(self):
        jobMarketplace = JobMarketplace()
        jobMarketplace.job_listings = [{"job_title": "Software Engineer", "company": "ABC Company", "requirements": ['skill1', 'skill2']}, {"job_title": "Software Engineer", "company": "ABC Company", "requirements": ['skill3', 'skill4']}]
        self.assertEqual(jobMarketplace.search_jobs("skill3"), [{'job_title': 'Software Engineer', 'company': 'ABC Company', 'requirements': ['skill3', 'skill4']}])

    def test_search_jobs_4(self):
        jobMarketplace = JobMarketplace()
        jobMarketplace.job_listings = [{"job_title": "Software Engineer", "company": "ABC Company", "requirements": ['skill1', 'skill2']}, {"job_title": "Software Engineer", "company": "ABC Company", "requirements": ['skill3', 'skill4']}]
        self.assertEqual(jobMarketplace.search_jobs("skill5"), [])

    def test_search_jobs_5(self):
        jobMarketplace = JobMarketplace()
        jobMarketplace.job_listings = [{"job_title": "Software Engineer", "company": "ABC Company", "requirements": ['skill1', 'skill2']}, {"job_title": "Software Engineer", "company": "ABC Company", "requirements": ['skill3', 'skill4']}]
        self.assertEqual(jobMarketplace.search_jobs("skill6"), [])

class JobMarketplaceTestGetJobApplicants(unittest.TestCase):
    def test_get_job_applicants(self):
        jobMarketplace = JobMarketplace()
        jobMarketplace.resumes = [{"name": "Tom", "skills": ['skill1', 'skill2'], "experience": "experience"}]
        jobMarketplace.job_listings = [{"job_title": "Software Engineer", "company": "ABC Company", "requirements": ['skill1', 'skill2']}]
        self.assertEqual(jobMarketplace.get_job_applicants(jobMarketplace.job_listings[0]), [{'name': 'Tom', 'skills': ['skill1', 'skill2'], 'experience': 'experience'}])

    def test_get_job_applicants_2(self):
        jobMarketplace = JobMarketplace()
        jobMarketplace.resumes = [{"name": "Tom", "skills": ['skill1', 'skill2'], "experience": "experience"}, {"name": "John", "skills": ['skill3', 'skill4'], "experience": "experience"}]
        jobMarketplace.job_listings = [{"job_title": "Software Engineer", "company": "ABC Company", "requirements": ['skill1', 'skill2']}]
        self.assertEqual(jobMarketplace.get_job_applicants(jobMarketplace.job_listings[0]), [{'name': 'Tom', 'skills': ['skill1', 'skill2'], 'experience': 'experience'}])

    def test_get_job_applicants_3(self):
        jobMarketplace = JobMarketplace()
        jobMarketplace.resumes = [{"name": "Tom", "skills": ['skill1', 'skill2'], "experience": "experience"}, {"name": "John", "skills": ['skill3', 'skill4'], "experience": "experience"}]
        jobMarketplace.job_listings = [{"job_title": "Software Engineer", "company": "ABC Company", "requirements": ['skill3', 'skill4']}]
        self.assertEqual(jobMarketplace.get_job_applicants(jobMarketplace.job_listings[0]), [{'name': 'John', 'skills': ['skill3', 'skill4'], 'experience': 'experience'}])

    def test_get_job_applicants_4(self):
        jobMarketplace = JobMarketplace()
        jobMarketplace.resumes = [{"name": "Tom", "skills": ['skill1', 'skill2'], "experience": "experience"}, {"name": "John", "skills": ['skill3', 'skill4'], "experience": "experience"}]
        jobMarketplace.job_listings = [{"job_title": "Software Engineer", "company": "ABC Company", "requirements": ['skill5', 'skill6']}]
        self.assertEqual(jobMarketplace.get_job_applicants(jobMarketplace.job_listings[0]), [])

    def test_get_job_applicants_5(self):
        jobMarketplace = JobMarketplace()
        jobMarketplace.resumes = [{"name": "Tom", "skills": ['skill1', 'skill2'], "experience": "experience"}, {"name": "John", "skills": ['skill3', 'skill4'], "experience": "experience"}]
        jobMarketplace.job_listings = [{"job_title": "Software Engineer", "company": "ABC Company", "requirements": ['skill6', 'skill7']}]
        self.assertEqual(jobMarketplace.get_job_applicants(jobMarketplace.job_listings[0]), [])

class JobMarketplaceTestMatchesRequirements(unittest.TestCase):
    def test_matches_requirements(self):
        jobMarketplace = JobMarketplace()
        self.assertEqual(jobMarketplace.matches_requirements({"name": "Tom", "skills": ['skill1', 'skill2'], "experience": "experience"}, ['skill1', 'skill2']), True)

    def test_matches_requirements_2(self):
        jobMarketplace = JobMarketplace()
        self.assertEqual(jobMarketplace.matches_requirements({"name": "Tom", "skills": ['skill1', 'skill2'], "experience": "experience"}, ['skill3', 'skill4']), False)

    def test_matches_requirements_3(self):
        jobMarketplace = JobMarketplace()
        self.assertEqual(jobMarketplace.matches_requirements({"name": "Tom", "skills": ['skill1', 'skill2'], "experience": "experience"}, ['skill5', 'skill6']), False)

    def test_matches_requirements_4(self):
        jobMarketplace = JobMarketplace()
        self.assertEqual(jobMarketplace.matches_requirements({"name": "Tom", "skills": ['skill1', 'skill2'], "experience": "experience"}, ['skill1', 'skill3']), False)

    def test_matches_requirements_5(self):
        jobMarketplace = JobMarketplace()
        self.assertEqual(jobMarketplace.matches_requirements({"name": "Tom", "skills": ['skill1', 'skill2'], "experience": "experience"}, ['skill1']), False)

class JobMarketplaceTestMain(unittest.TestCase):
    def test_main(self):
        jobMarketplace = JobMarketplace()
        jobMarketplace.post_job("Software Engineer", "ABC Company", ['skill1', 'skill2'])
        jobMarketplace.post_job("Mechanical Engineer", "XYZ Company", ['skill3', 'skill4'])
        self.assertEqual(jobMarketplace.job_listings, [{'job_title': 'Software Engineer', 'company': 'ABC Company', 'requirements': ['skill1', 'skill2']}, {'job_title': 'Mechanical Engineer', 'company': 'XYZ Company', 'requirements': ['skill3', 'skill4']}])
        jobMarketplace.remove_job(jobMarketplace.job_listings[1])
        self.assertEqual(jobMarketplace.job_listings, [{'job_title': 'Software Engineer', 'company': 'ABC Company', 'requirements': ['skill1', 'skill2']}])
        jobMarketplace.submit_resume("Tom", ['skill1', 'skill2'], "experience")
        self.assertEqual(jobMarketplace.resumes, [{'name': 'Tom', 'skills': ['skill1', 'skill2'], 'experience': 'experience'}])
        jobMarketplace.withdraw_resume(jobMarketplace.resumes[0])
        self.assertEqual(jobMarketplace.resumes, [])
        self.assertEqual(jobMarketplace.search_jobs("skill1"), [{'job_title': 'Software Engineer', 'company': 'ABC Company', 'requirements': ['skill1', 'skill2']}])
        jobMarketplace.resumes = [{"name": "Tom", "skills": ['skill1', 'skill2'], "experience": "experience"}]
        self.assertEqual(jobMarketplace.get_job_applicants(jobMarketplace.job_listings[0]), [{'name': 'Tom', 'skills': ['skill1', 'skill2'], 'experience': 'experience'}])
        self.assertEqual(jobMarketplace.matches_requirements({"name": "Tom", "skills": ['skill1', 'skill2'], "experience": "experience"}, ['skill1', 'skill2']), True)

