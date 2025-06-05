from donut_charts import donuts

def main():
    results = {
        'strong': [36,11,3,0,0],
        'medium':[13,11,4,12,10],
        'weak':[10,10,10,10,10]
    }
    donuts(results, "sampled")
    
main()