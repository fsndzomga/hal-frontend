def get_fallback_accuracy(results):
    if 'accuracy' in results and results['accuracy'] is not None:
        return results['accuracy']
    elif 'average_correctness' in results and results['average_correctness'] is not None:
        return results['average_correctness']
    elif 'success_rate' in results and results['success_rate'] is not None:
        return results['success_rate']
    elif 'average_score' in results and results['average_score'] is not None:
        return results['average_score']
    else:
        return None