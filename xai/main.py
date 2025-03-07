from cf_dataset import *
from tcga_dataset import *
from gene_analysis import *

if __name__ == "__main__":

    ### CF attribution values
    # run all using model aggregation
    #run_cf_analysis(beta=1, bar_plot=True, venn_diagram=True)

    # run individual XAI method analysis
    #xai_method = "deepliftshap"
    #top_features, all_features, all_attr = disease_results_using_model_aggr(xai_method, barplot=True, beta=0.01)

    # check only CFTR rank
    #get_cftr_rank(all_features)

    ## load precomputed feature rankings
    dls_features = load_sorted_features("deepliftshap")
    lime_features = load_sorted_features("lime")
    ig_features = load_sorted_features("integrated_gradients")

    top_100_dls = dls_features[:100]
    top_100_lime = lime_features[:100]
    top_100_ig = ig_features[:100]

    # print overlapping features and venn diagram
    print("\n🔹 Overlap of Top 100 Features:")
    feature_overlap(top_100_dls, top_100_lime, top_100_ig, dataset="cf")
    plot_venn_diagram(top_100_dls, top_100_lime, top_100_ig, beta=1, n=100, show=True, dataset="cf")

    ### TCGA attribution values
    ## run tcga for LAML, PRAD, THCA
    # get_results_final_cancers(beta=0.01)

    # # specify cancer and beta
    # cancer_type = "LAML"
    # beta = 0.01

    # all analysis
    #run_tcga_analysis(cancer_type=cancer_type, beta=beta, bar_plot=True, venn_diagram=True, show=False)

    # # load precomputed feature rankings
    # dls_features = load_cancer_features("deepliftshap", cancer=cancer_type, beta=beta)
    # lime_features = load_cancer_features("lime", cancer=cancer_type, beta=beta)
    # ig_features = load_cancer_features("integrated_gradients", cancer=cancer_type, beta=beta)

    # # Take the top 100 features from each method
    # top_100_dls = dls_features[:100]
    # top_100_lime = lime_features[:100]
    # top_100_ig = ig_features[:100]

    # # print overlapping features and venn diagram
    # print("\nOverlap of Top 100 Features:")
    # feature_overlap(top_100_dls, top_100_lime, top_100_ig, dataset="tcga")
    # plot_venn_diagram(top_100_dls, top_100_lime, top_100_ig, beta=beta, n=100, show=True, dataset="tcga",
    #                   cancer_type=cancer_type)

    # # statistical rank distribution analysis
    # df_all_methods = parse_all_methods(cancer_type, beta=beta)
    # stats = rank_distribution_stats(df_all_methods)
    # print(stats)

    ### Biological Evaluation
    beta_values = [0.01, 1]
    gene_n = 15

    # tcga_cancers = {}
    # for cancer in cancer_types:
    #     dls_features = load_cancer_features("deepliftshap", cancer=cancer, beta="001")
    #     lime_features = load_cancer_features("lime", cancer=cancer, beta="001")
    #     ig_features = load_cancer_features("integrated_gradients", cancer=cancer, beta="001")
    #
    #     # combine unique genes from all methods
    #     combined_gene_list = list(set(dls_features + lime_features + ig_features))
    #     tcga_cancers[cancer] = combined_gene_list


    # compute gene ranks and stat testing
    # for cancer, genes in tcga_cancers.items():
    #     print(f"\n🔹 {cancer} Gene Ranking:")
    #     for beta in beta_values:
    #         print(f"\nBeta = {beta}:")
    #         get_all_gene_ranks_tcga(cancer, beta, genes)
    #
    #     print(f"\n🔹 {cancer} Paired T-test Between Beta Values:")
    #     pairwise_test_ranks(cancer, genes)

    # # CF functional gene analysis
    # print("\n🔹 CF Functional Analysis")
    # for beta in beta_values:
    #     print(f"\nBeta = {beta}:")
    #     get_gene_info_cf(beta=beta, gene_n=gene_n)
    #
    # # TCGA functional gene analysis
    # for cancer in tcga_cancers.keys():
    #     print(f"\n🔹 {cancer} Functional Analysis")
    #     for beta in beta_values:
    #         print(f"\nBeta = {beta}:")
    #         get_gene_info_tcga(beta=beta, cancer_type=cancer, gene_n=gene_n)
