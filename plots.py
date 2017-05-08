import matplotlib.pyplot as plt

def plots():
    data = reader_process('risk_metric_p20_d6-20_n50')
    recovery_err_nuc = defaultdict(list)
    recovery_err_L12 = defaultdict(list)

    norm_nuc = defaultdict(list)
    norm_L12 = defaultdict(list)
    start = 5000
    for run in data:
        keys = ['rel_err_list', 'n', 'd', 'p', 'step']
        risks, n, d, p, step = [run[k] for k in keys]
        tries_risk_nuc = 5000
        tries_risk_L12 = 5000
        for r in risks:
            if r[0] > .003:
                tries_risk_nuc += step
            if r[1] > .003:
                tries_risk_L12 += step
        print 'd', d, tries_risk_nuc, tries_risk_L12
        norm_nuc[d].append(tries_risk_nuc)
        norm_L12[d].append(tries_risk_L12)

        # Samples to recovery error
        recovery_samples_nuc = 5000
        recovery_samples_L12 = 5000
        Ktrue = np.eye(p)
        for i in range(d, p):
            Ktrue[i, i] = 0

        for K in run['Ks'][1:]:
            nuc_recovery = np.linalg.norm(
                Ktrue - K[0], 'fro')**2 / np.linalg.norm(Ktrue, 'fro')**2
            L12_recovery = np.linalg.norm(
                Ktrue - K[1], 'fro')**2 / np.linalg.norm(Ktrue, 'fro')**2
            if nuc_recovery > .10:
                recovery_samples_nuc += step
            if L12_recovery > .10:
                recovery_samples_L12 += step

        recovery_err_nuc[d].append(recovery_samples_nuc)
        recovery_err_L12[d].append(recovery_samples_L12)
    x = sorted(norm_nuc.keys())
    print x
    plt.figure(1)
    plt.subplot(211)
    plt.title('Samples to get excess_risk < .003')
    plt.errorbar(x, [np.mean(norm_nuc[d]) for d in x], yerr=[
                 np.std(norm_nuc[d]) for d in x], color='red')
    plt.errorbar(x, [np.mean(norm_L12[d])
                     for d in x], yerr=[np.std(norm_L12[d]) for d in x])
    plt.subplot(212)
    plt.title('Samples to get generalization error < .01')
    plt.errorbar(x, [np.mean(recovery_err_nuc[d]) for d in x], yerr=[
                 np.std(recovery_err_nuc[d]) for d in x], color='red')
    plt.errorbar(x, [np.mean(recovery_err_L12[d])
                     for d in x], yerr=[np.std(recovery_err_L12[d]) for d in x])
    plt.show()
