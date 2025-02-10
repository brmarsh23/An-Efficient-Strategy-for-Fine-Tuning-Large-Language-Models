kubectl delete -k deploy/manifest/overlay/tenzingprod/model-training/
git pull
kubectl apply -k deploy/manifest/overlay/tenzingprod/model-training/