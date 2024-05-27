import gradio as gr
import torch
import torch.nn.functional as F
from sklearn.neighbors import NearestNeighbors
from dataset.dataloader import arxiv_dataset, load_titleabs
from model.bert import BertNode2Vec
from model.scibert import SciBertNode2Vec
from utils.args import get_app_args
from app.get_abstract import title_to_abs

# Initialize arguments and model
args = get_app_args()
device = args.device
data = arxiv_dataset()
titleabs = load_titleabs()

# Load pre-trained embeddings
with torch.no_grad():
    if args.model_type == 'bert':
        model = BertNode2Vec(device=device)
        model.load(args.pretrain, device)
        emb = model.embed_all(data)
    elif args.model_type == 'scibert':
        model = SciBertNode2Vec(device=device)
        model.load(args.pretrain, device)
        emb = model.embed_all(data)

F.normalize(emb, p=2, dim=1)

knn = NearestNeighbors(n_neighbors=args.k, metric='cosine')
knn.fit(emb.numpy())

# Define the function for recommendation
def recommend_paper(title):
    try:
        abstract = title_to_abs(title)
    except Exception as e:
        return str(e), []

    if not abstract:
        return "Abstract not found for the given title.", []

    inf_emb = model.inference(abstract).cpu()
    inf_emb = F.normalize(inf_emb, p=2, dim=1)
    
    distances, nearest_indices = knn.kneighbors(inf_emb.numpy(), return_distance=True)

    recommendations = []
    for i, idx in enumerate(nearest_indices[0]):
        recommendations.append({
            'Title': titleabs['title'][idx],
            'Abstract': titleabs['abs'][idx],
            'Distance': str(distances[0][i])
        }.values())

    return "Success", recommendations

if __name__ == "__main__":
    # Define the Gradio interface
    title_input = gr.Textbox(lines=1, placeholder="Enter the article title")
    output_text = gr.Textbox()
    output_recommendations = gr.Dataframe(headers=["Title", "Abstract", "Distance"])

    demo = gr.Interface(
        fn=recommend_paper, 
        inputs=title_input, 
        outputs=[
            gr.Row([output_text]),
            gr.Row([output_recommendations])
        ], 
        title="Research Paper Recommendation System", 
        description="Enter the title of a research paper to get recommendations based on its abstract."
    )

    demo.launch(share=True)