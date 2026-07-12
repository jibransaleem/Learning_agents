import os
import textwrap
from collections import Counter
from typing import TypedDict, Annotated, Optional

import requests
from dotenv import load_dotenv
from pydantic import BaseModel, Field

from langchain_core.messages import HumanMessage
from langchain_groq import ChatGroq

from langgraph.graph import START, END, StateGraph
from langgraph.graph.message import add_messages
from langgraph.types import RetryPolicy
from langgraph.checkpoint.memory import InMemorySaver

load_dotenv()

BASE_URL = "https://api.github.com"


def load_model():
    return ChatGroq(model="llama-3.3-70b-versatile", api_key=os.getenv("GROQ_API"))


# ============================================================
# State
# ============================================================
class CardState(TypedDict):
    messages: Annotated[list, add_messages]
    user: str
    tone: str
    github_token: Optional[str]
    profile_raw: dict
    card_data: str
    image_prompt: str
    image_path: str


# ============================================================
# GitHub data fetch
# ============================================================
def get_github_profile(username: str, token: str | None = None) -> dict:
    """
    Fetch comprehensive public GitHub profile information.

    Args:
        username: GitHub username
        token: Optional GitHub Personal Access Token

    Returns:
        Dictionary containing profile, repositories, languages, organizations,
        recent events, and profile README.
    """
    headers = {"Accept": "application/vnd.github+json"}
    if token:
        headers["Authorization"] = f"Bearer {token}"

    def get(url):
        response = requests.get(url, headers=headers, timeout=15)
        response.raise_for_status()
        return response.json()

    # ----------------------------
    # Basic Profile
    # ----------------------------
    profile = get(f"{BASE_URL}/users/{username}")

    # ----------------------------
    # Public Repositories
    # ----------------------------
    repos = get(f"{BASE_URL}/users/{username}/repos?per_page=100")

    repo_data = []
    language_counter = Counter()
    topic_counter = Counter()

    for repo in repos:
        # Languages for each repo
        try:
            langs = get(repo["languages_url"])
            language_counter.update(langs)
        except Exception:
            pass

        topic_counter.update(repo.get("topics", []))

        repo_data.append({
            "name": repo["name"],
            "description": repo["description"],
            "language": repo["language"],
            "stars": repo["stargazers_count"],
            "forks": repo["forks_count"],
            "topics": repo.get("topics", []),
            "created_at": repo["created_at"],
            "updated_at": repo["updated_at"],
            "url": repo["html_url"],
        })

    # ----------------------------
    # Organizations
    # ----------------------------
    try:
        orgs = get(f"{BASE_URL}/users/{username}/orgs")
        organizations = [o["login"] for o in orgs]
    except Exception:
        organizations = []

    # ----------------------------
    # Public Events
    # ----------------------------
    try:
        events = get(f"{BASE_URL}/users/{username}/events/public")
        recent_events = [
            {"type": e["type"], "repo": e["repo"]["name"], "created_at": e["created_at"]}
            for e in events[:10]
        ]
    except Exception:
        recent_events = []

    # ----------------------------
    # Profile README
    # ----------------------------
    try:
        readme = get(f"{BASE_URL}/repos/{username}/{username}/readme")
        profile_readme = readme.get("download_url")
    except Exception:
        profile_readme = None

    # ----------------------------
    # Sort repositories
    # ----------------------------
    top_repositories = sorted(repo_data, key=lambda x: x["stars"], reverse=True)[:10]

    return {
        "profile": {
            "username": profile["login"],
            "name": profile.get("name"),
            "bio": profile.get("bio"),
            "company": profile.get("company"),
            "location": profile.get("location"),
            "blog": profile.get("blog"),
            "twitter": profile.get("twitter_username"),
            "followers": profile["followers"],
            "following": profile["following"],
            "public_repos": profile["public_repos"],
            "public_gists": profile["public_gists"],
            "created_at": profile["created_at"],
            "avatar": profile["avatar_url"],
            "profile_url": profile["html_url"],
        },
        "repositories": repo_data,
        "top_repositories": top_repositories,
        "languages": dict(language_counter.most_common()),
        "topics": dict(topic_counter.most_common()),
        "organizations": organizations,
        "recent_events": recent_events,
        "profile_readme": profile_readme,
    }


# ============================================================
# Structured-output schemas
# ============================================================
class RequestInfo(BaseModel):
    username: str = Field(
        description="The GitHub username mentioned in the request. Empty string if none was given."
    )
    tone: str = Field(
        default="professional",
        description="The tone/style requested for the card (e.g. 'professional', 'playful', "
                    "'minimal', 'cyberpunk'). Default to 'professional' if not mentioned.",
    )


class ParseData(BaseModel):
    out: str = Field(
        description=(
            "Summarize the provided GitHub profile into a profile-card friendly "
            "format (100-200 words). Include: who the developer is, their primary "
            "tech stack, strongest programming languages, key interests, notable "
            "projects, activity level, and any noteworthy achievements. Focus on "
            "the most impactful information only. Keep the tone professional, "
            "engaging, and factual. Do not fabricate details or mention missing "
            "information."
        )
    )


class ImagePrompt(BaseModel):
    prompt: str = Field(
        description="A short, vivid text-to-image prompt (under 50 words) for a background/banner "
                    "image that visually represents this developer's profile, matching the "
                    "requested tone. No text or letters in the image."
    )


# ============================================================
# Graph nodes
# ============================================================
def extract_request(state: CardState) -> dict:
    """Parse the user's message to get the GitHub username and desired card tone."""
    model = load_model()
    model_ = model.with_structured_output(RequestInfo)

    last_message = state["messages"][-1].content
    parsed = model_.invoke(
        f"Extract the GitHub username and desired card tone from this request:\n{last_message}"
    )

    if not parsed.username:
        raise ValueError("No GitHub username found in the request.")

    return {"user": parsed.username, "tone": parsed.tone}


def profile_data(state: CardState) -> dict:
    """Fetch the GitHub profile and summarize it for the card."""
    user_name = state["user"]
    token = state.get("github_token") or os.getenv("GITHUB_TOKEN")
    data = get_github_profile(username=user_name, token=token)

    model = load_model()
    model_ = model.with_structured_output(ParseData)

    summary = model_.invoke(
        f"Here is raw GitHub profile data as JSON:\n{data}\n\n"
        f"Write the profile-card summary as instructed."
    )

    return {
        "profile_raw": data,
        "card_data": summary.out,
    }


def build_image_prompt(state: CardState) -> dict:
    """Turn the profile summary into a background-image prompt."""
    model = load_model()
    model_ = model.with_structured_output(ImagePrompt)
    result = model_.invoke(
    f"""
    You are an expert AI image prompt engineer.

    Developer Profile:
    {state["card_data"]}

    Tone: {state.get("tone", "professional")}

    Generate ONE highly detailed prompt for an AI image model (FLUX, SDXL, GPT Image).

    The image should look like a premium football/FIFA/EA FC Ultimate Team player card, but redesigned for a software developer.

    Requirements:
    - Vertical collectible card.
    - Premium holographic frame with futuristic UI.
    - Cyberpunk + glassmorphism aesthetic.
    - Represent the developer's skills through icons and visual elements.
    - Add technology-inspired effects (AI networks, code streams, circuit traces, cloud nodes, neural networks, terminal windows, data particles).
    - Display skill badges and rating placeholders similar to football cards.
    - Include decorative stat panels (leave space for later text overlay).
    - Rich metallic gold/silver accents depending on expertise.
    - Dynamic lighting, volumetric glow, reflections and depth.
    - No text, no numbers, no logos, no company names, no watermarks.
    - No human face or portrait.
    - Clean center area reserved for future profile information.
    - High-end gaming collectible aesthetic.
    - Ultra detailed, 3D, cinematic, Unreal Engine quality, octane render, 8k.

    Return ONLY the image prompt.
    """
    )
    return {"image_prompt": result.prompt}


def generate_card_image(state: CardState) -> dict:
    """Generate the background banner image via Hugging Face Inference."""
    from huggingface_hub import InferenceClient

    client = InferenceClient(
        provider="hf-inference",
        api_key=os.getenv("HF_TOKEN"),
    )

    image = client.text_to_image(
        prompt=state["image_prompt"],
        model="black-forest-labs/FLUX.1-schnell",
    )

    bg_path = f"{state['user']}_card_bg.png"
    image.save(bg_path)
    return {"image_path": bg_path}


def render_card(state: CardState) -> dict:
    """Composite the summary text over the generated background into the final card."""
    from PIL import Image, ImageDraw, ImageFont

    bg = Image.open(state["image_path"]).convert("RGBA").resize((1024, 512))
    overlay = Image.new("RGBA", bg.size, (0, 0, 0, 140))
    card = Image.alpha_composite(bg, overlay)
    draw = ImageDraw.Draw(card)

    try:
        font_title = ImageFont.truetype("arialbd.ttf", 36)
        font_body = ImageFont.truetype("arial.ttf", 20)
    except OSError:
        font_title = ImageFont.load_default()
        font_body = ImageFont.load_default()

    profile = state["profile_raw"]["profile"]
    draw.text((40, 30), profile.get("name") or profile["username"], font=font_title, fill="white")

    wrapped = textwrap.fill(state["card_data"], width=90)
    draw.multiline_text((40, 90), wrapped, font=font_body, fill="white", spacing=6)

    final_path = f"{state['user']}_github_card.png"
    card.convert("RGB").save(final_path)
    return {"image_path": final_path}


# ============================================================
# Graph assembly
# ============================================================
network_retry = RetryPolicy(max_attempts=3)

graph = StateGraph(CardState)
graph.add_node("extract_request", extract_request)
graph.add_node("profile_data", profile_data, retry_policy=network_retry)
graph.add_node("build_image_prompt", build_image_prompt)
graph.add_node("generate_card_image", generate_card_image, retry_policy=network_retry)
graph.add_node("render_card", render_card)

graph.add_edge(START, "extract_request")
graph.add_edge("extract_request", "profile_data")
graph.add_edge("profile_data", "build_image_prompt")
graph.add_edge("build_image_prompt", "generate_card_image")
graph.add_edge("generate_card_image", "render_card")
graph.add_edge("render_card", END)

app = graph.compile(checkpointer=InMemorySaver())


if __name__ == "__main__":
    config = {"configurable": {"thread_id": "github-card-1"}}
    result = app.invoke(
        {"messages": [HumanMessage(content="Make me a professional GitHub card for jibransaleem like a football score card")]},
        config=config,
    )
    print(f"Card saved to: {result['image_path']}")
    print(result["card_data"])