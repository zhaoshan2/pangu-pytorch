"""
Pseudocode for converting the onnx weights to torch weights
"""
# Match between onnx key and torch key
lookUpTable = os.path.join(PATH, 'keys_all.csv')
lookUpTable = pd.read_csv(lookUpTable)
# Load onnx file of pretrained pangu model
model_24 = onnx.load(cfg.PG.BENCHMARK.PRETRAIN_24)

graph = model_24.graph
INTIALIZERS = model_24.graph.initializer
onnx_weights = {}
for initializer in INTIALIZERS:
     W = np_helper.to_array(initializer)
     onnx_weights[initializer.name] = W
# Pangu model in pytorch
model = PanguModel(device=device).to(device)

torch_keys = lookUpTable["torch_name"]

count = 0
print("Load pretrain with bias")
# Load the onnx weight to pangu model layer by layer
for name, param in model.named_parameters():
    if param.requires_grad:
    #    print(count, name)
    #    print(torch_keys[count])

       row = lookUpTable[lookUpTable['torch_name'] == name]
       if row.empty:
           print("no record torch key ", name)
       onnx_name = row['onnx_name'].values[0]
       if isinstance(onnx_name, str):
          w  = torch.tensor(onnx_weights[onnx_name])
        #   print("shapes", param.data.shape, w.shape)
          if len(param.data.shape) == 1:
              assert param.data.shape == w.shape
              param.data = w.clone().to(device)
              param.requires_grad = False

          elif len(param.data.shape) == 2:
              assert param.data.shape == w.T.shape
              param.data = w.T.clone().to(device)
              param.requires_grad = False
          elif len(param.data.shape) == 3:
              assert param.data.shape == w.shape
              param.data = w.clone().to(device)
              param.requires_grad = False
          elif len(param.data.shape) == 5:
              assert param.data.shape == w.shape
              param.data = w.clone().to(device)
              param.requires_grad = False
# Save the torch weights
torch.save(model,os.path.join(output_path,"onnx2torch.pth"))
