from django.shortcuts import render, redirect
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from io import BytesIO
import base64
import os

# -------------------------------------------------
# FUNGSI KMEANS MANUAL
# -------------------------------------------------
def kmeans(X, k, max_iters=300, random_state=0):
    np.random.seed(random_state)
    centroids = X[np.random.choice(X.shape[0], k, replace=False)]

    for _ in range(max_iters):
        distances = np.linalg.norm(X[:, None] - centroids[None, :], axis=2)
        labels = np.argmin(distances, axis=1)

        new_centroids = np.array([
            X[labels == j].mean(axis=0) if np.any(labels == j) else centroids[j]
            for j in range(k)
        ])

        if np.allclose(centroids, new_centroids):
            break

        centroids = new_centroids

    wcss = np.sum((X - centroids[labels]) ** 2)
    return labels, centroids, wcss


# =================================================
# 1️⃣ INPUT JUMLAH K
# =================================================
def cluster_input(request):
    if request.method == "POST":
        request.session["k"] = int(request.POST.get("k"))
        return redirect("proses_kmeans")

    return render(request, 'clustering/menentukan_cluster.html')


# =================================================
# 2️⃣ PROSES KMEANS
# =================================================
def proses_kmeans(request):
    BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    df = pd.read_csv(os.path.join(BASE_DIR, 'clustering', 'Mall_Customers.csv'))

    k = int(request.session.get("k", 3))

    # Standardisasi nama kolom
    df = df.rename(columns={
        "Age": "age",
        "Annual Income (k$)": "income",
        "Spending Score (1-100)": "spending"
    })

    X = df[['age', 'income', 'spending']].values

    # Hitung centroid awal (manual)
    np.random.seed(0)
    initial_centroids = X[np.random.choice(X.shape[0], k, replace=False)]

    # Hitung jarak
    distances = np.linalg.norm(X[:, None] - initial_centroids[None, :], axis=2)
    labels = np.argmin(distances, axis=1)

    distance_df = pd.DataFrame(distances, columns=[f"C{i}" for i in range(k)])

    centroid_awal = pd.DataFrame(
        initial_centroids, columns=['age', 'income', 'spending']
    )

    centroid_akhir = pd.DataFrame(
        [X[labels == i].mean(axis=0) for i in range(k)],
        columns=['age', 'income', 'spending']
    )

    df["cluster"] = labels

    context = {
        "k": k,
        "data_asli": df.head(200).to_html(classes="table table-bordered"),
        "distance_table": distance_df.to_html(classes="table table-bordered table-sm"),
        "centroid_awal": centroid_awal.to_html(classes="table table-bordered"),
        "centroid_akhir": centroid_akhir.to_html(classes="table table-bordered"),
        "cluster_hasil": df[['age', 'income', 'spending', 'cluster']].head(200)
                        .to_html(classes="table table-bordered"),
    }
    return render(request, "clustering/proses_kmeans.html", context)


# =================================================
# 3️⃣ METODE ELBOW
# =================================================
def elbow(request):
    BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    df = pd.read_csv(os.path.join(BASE_DIR, 'clustering', 'Mall_Customers.csv'))

    df = df.rename(columns={
        "Age": "age",
        "Annual Income (k$)": "income",
        "Spending Score (1-100)": "spending"
    })

    X = df[['age', 'income', 'spending']].values

    wcss_list = []
    for i in range(1, 11):
        _, _, wcss = kmeans(X, i)
        wcss_list.append(wcss)

    fig, ax = plt.subplots()
    ax.plot(range(1, 11), wcss_list, marker='o')
    ax.set_title("Metode Elbow")
    ax.set_xlabel("Jumlah Cluster")
    ax.set_ylabel("WCSS")

    buffer = BytesIO()
    plt.savefig(buffer, format="png")
    elbow_plot = base64.b64encode(buffer.getvalue()).decode()
    plt.close()

    return render(request, 'clustering/elbow.html', {'elbow_plot': elbow_plot})


# =================================================
# 4️⃣ HALAMAN UTAMA CLUSTERING
# =================================================
def clustering(request):
    BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    df = pd.read_csv(os.path.join(BASE_DIR, 'clustering', 'Mall_Customers.csv'))

    # Rename kolom agar aman
    df = df.rename(columns={
        "Age": "age",
        "Annual Income (k$)": "income",
        "Spending Score (1-100)": "spending"
    })

    X = df[['age', 'income', 'spending']].values

    k = int(request.session.get("k", 3))

    labels, centroids, wcss = kmeans(X, k)
    df["cluster"] = labels

    # Kelompokkan data per cluster
    cluster_groups = []
    for i in range(k):
        cluster_groups.append({
            "id": i,
            "data": df[df["cluster"] == i].to_dict("records")
        })

    # Plot cluster
    fig, ax = plt.subplots()
    for i in range(k):
        ax.scatter(X[labels == i, 0], X[labels == i, 2], label=f"Cluster {i}")

    ax.scatter(centroids[:, 0], centroids[:, 2], s=200, c='black', marker='X')
    ax.set_title("Visualisasi Cluster")
    ax.set_xlabel("Age")
    ax.set_ylabel("Spending")
    ax.legend()

    buffer = BytesIO()
    plt.savefig(buffer, format="png")
    cluster_plot = base64.b64encode(buffer.getvalue()).decode()
    plt.close()

    summary = df.groupby("cluster")[['age', 'income', 'spending']].mean().astype(int)

    return render(request, 'clustering/clustering.html', {
        'cluster_plot': cluster_plot,
        'summary': summary.to_html(classes="table table-bordered"),
        'clusters': cluster_groups,
        'k': k,
    })


# =================================================
# 5️⃣ KESIMPULAN
# =================================================
def kesimpulan(request):
    return render(request, 'clustering/kesimpulan.html', {
        'kesimpulan_text': "Analisis K-Means telah berhasil dilakukan."
    })
