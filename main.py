from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import os
from dotenv import load_dotenv
from neo4j import GraphDatabase
from langchain_community.embeddings import OpenAIEmbeddings
from langchain_community.chat_models import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from fastapi.middleware.cors import CORSMiddleware
from datetime import date
from typing import List, Optional
from uuid import uuid4

# Load environment variables
load_dotenv()

app = FastAPI(
    title="Financial Advisor API",
    description="API for financial advice combining user data with financial knowledge",
    version="1.0.0"
)

# CORS configuration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Adjust this in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ========== MODELS ==========
class QuestionRequest(BaseModel):
    email: str
    question: str

class Savings(BaseModel):
    savings_name: str
    category: str
    amount: float
    date_created: date

class FinancialGoal(BaseModel):
    category: str
    goal_name: str
    goal_description: str
    target_amount: float
    deadline: date

class Expenditure(BaseModel):
    category: str
    amount: float
    date: date

class UserCreate(BaseModel):
    username: str
    email: str
    phone: str
    occupation: str
    dob: date
    monthly_income: float
    password: str
    goals: Optional[List[FinancialGoal]] = None
    savings: Optional[List[Savings]] = None
    expenditures: Optional[List[Expenditure]] = None

class UserResponse(BaseModel):
    id: str
    username: str
    email: str
    phone: str
    occupation: str
    dob: date
    monthly_income: float
    goals: List[FinancialGoal]
    savings: List[Savings]
    expenditures: List[Expenditure]

# ========== DATABASE CLASS ==========
class Neo4jDatabase:
    def __init__(self):
        self.driver = GraphDatabase.driver(
            os.getenv("NEO4J_URI"),
            auth=(
                os.getenv("NEO4J_USERNAME"),
                os.getenv("NEO4J_PASSWORD")
            )
        )

    def close(self):
        self.driver.close()

    def create_user(self, user_data):
        query = """
        CREATE (u:User {
            id: $id,
            username: $username,
            email: $email,
            phone: $phone,
            occupation: $occupation,
            dob: $dob,
            monthly_income: $monthly_income,
            password: $password
        })
        WITH u
        
        FOREACH (goal IN $goals |
            CREATE (g:Goal {
                id: randomUUID(),
                category: goal.category,
                name: goal.goal_name,
                description: goal.goal_description,
                target_amount: goal.target_amount,
                deadline: goal.deadline
            })
            CREATE (u)-[:HAS_GOAL]->(g)
        )
        
        FOREACH (saving IN $savings |
            CREATE (s:Saving {
                id: randomUUID(),
                name: saving.savings_name,
                category: saving.category,
                amount: saving.amount,
                date_created: saving.date_created
            })
            CREATE (u)-[:HAS_SAVING]->(s)
        )
        
        FOREACH (expense IN $expenditures |
            CREATE (e:Expense {
                id: randomUUID(),
                category: expense.category,
                amount: expense.amount,
                date: expense.date
            })
            CREATE (u)-[:HAS_EXPENSE]->(e)
        )
        
        WITH u
        OPTIONAL MATCH (u)-[:HAS_GOAL]->(g:Goal)
        OPTIONAL MATCH (u)-[:HAS_SAVING]->(s:Saving)
        OPTIONAL MATCH (u)-[:HAS_EXPENSE]->(e:Expense)
        RETURN u, 
            COLLECT(DISTINCT g) AS goals, 
            COLLECT(DISTINCT s) AS savings,
            COLLECT(DISTINCT e) AS expenses
        """
        with self.driver.session() as session:
            result = session.run(query, id=str(uuid4()), **user_data)
            return result.single()

    def get_user_by_email(self, email):
        query = """
        MATCH (u:User {email: $email})
        OPTIONAL MATCH (u)-[:HAS_GOAL]->(g:Goal)
        OPTIONAL MATCH (u)-[:HAS_SAVING]->(s:Saving)
        OPTIONAL MATCH (u)-[:HAS_EXPENSE]->(e:Expense)
        RETURN u, 
               COLLECT(DISTINCT g) AS goals, 
               COLLECT(DISTINCT s) AS savings,
               COLLECT(DISTINCT e) AS expenses
        """
        with self.driver.session() as session:
            result = session.run(query, email=email)
            return result.single()

# ========== FINANCIAL ADVISOR CLASS ==========
class FinancialAdvisor:
    def __init__(self, db: Neo4jDatabase):
        self.db = db
        self.embeddings = OpenAIEmbeddings()
        self.llm = ChatOpenAI(model="gpt-4", temperature=0.3)
        self.index_name = "financial-knowledge"
        self.prompt_template = ChatPromptTemplate.from_messages([
            ("system", """You are a FINANCIAL ADVISOR. If a question is not financial:
            1. Say "I specialize in financial advice" 
            2. Politely decline to answer
            3. Suggest rephrasing as financial question
            
            For FINANCIAL questions, combine:
            1. GENERAL KNOWLEDGE: {pdf_context}
            2. USER DATA: {user_data}
            Provide specific financial advice."""),
            ("human", "{question}")
        ])

    def get_user_data(self, email):
        user_data = self.db.get_user_by_email(email)
        if not user_data:
            return {"error": "User not found"}
        return self._format_user_data(user_data)

    def _format_user_data(self, user_record):
        user = user_record["u"]
        return {
            "name": user["username"],
            "incomes": [{"amount": user["monthly_income"], "type": "monthly"}],
            "goals": [{
                "name": g["name"],
                "target": g["target_amount"],
                "deadline": g["deadline"]
            } for g in user_record["goals"]],
            "expenses": [{
                "category": e["category"],
                "amount": e["amount"]
            } for e in user_record["expenses"]]
        }

    def query_knowledge(self, question):
        from pinecone import Pinecone
        pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))
        index = pc.Index(self.index_name)
        
        query_embedding = self.embeddings.embed_query(question)
        response = index.query(
            vector=query_embedding,
            top_k=3,
            include_metadata=True,
            namespace="financial_docs",
            filter={"is_financial": {"$eq": True}}
        )
        
        matches = response.get('matches', [])
        return [match.metadata["text"] for match in matches] if matches else ["No relevant financial information found."]

    def is_financial_question(self, question):
        FINANCIAL_KEYWORDS = {
            'invest', 'stock', 'bond', 'retirement', 'savings', 
            'tax', 'mortgage', 'loan', 'interest', 'portfolio',
            'financial', 'money', 'wealth', 'income', 'expense'
        }
        question_lower = question.lower()
        return any(keyword in question_lower for keyword in FINANCIAL_KEYWORDS)

    def ask(self, email, question):
        if not self.is_financial_question(question):
            return "I specialize in financial advice. Please ask about investments, taxes, retirement, or other money-related topics."
        
        user_data = self.get_user_data(email)
        if "error" in user_data:
            return f"Error fetching user data: {user_data['error']}"
        
        pdf_context = self.query_knowledge(question)
        prompt = self.prompt_template.format_messages(
            pdf_context="\n\n".join(pdf_context),
            user_data=str(user_data),
            question=question
        )
        return self.llm.invoke(prompt).content

# ========== INITIALIZE SERVICES ==========
db = Neo4jDatabase()
advisor = FinancialAdvisor(db)

# ========== API ENDPOINTS ==========
@app.post("/register/", response_model=UserResponse)
async def register_user(user: UserCreate):
    try:
        user_data = user.model_dump()
        user_data["dob"] = user_data["dob"].isoformat()
        
        if user.goals:
            for goal in user_data["goals"]:
                goal["deadline"] = goal["deadline"].isoformat()
        
        if user.savings:
            for saving in user_data["savings"]:
                saving["date_created"] = saving["date_created"].isoformat()
        
        if user.expenditures:
            for expense in user_data["expenditures"]:
                expense["date"] = expense["date"].isoformat()
        
        created_user = db.create_user(user_data)
        if not created_user:
            raise HTTPException(status_code=400, detail="Failed to create user")
            
        return format_user_response(created_user)
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/users/{email}", response_model=UserResponse)
async def get_user(email: str):
    user_record = db.get_user_by_email(email)
    if not user_record:
        raise HTTPException(status_code=404, detail="User not found")
    return format_user_response(user_record)

@app.post("/ask/")
async def ask_question(request: QuestionRequest):
    try:
        response = advisor.ask(request.email, request.question)
        return {"response": response}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# ========== HELPER FUNCTIONS ==========
def format_user_response(user_record):
    user = user_record["u"]
    return {
        "id": user["id"],
        "username": user["username"],
        "email": user["email"],
        "phone": user["phone"],
        "occupation": user["occupation"],
        "dob": date.fromisoformat(user["dob"]),
        "monthly_income": user["monthly_income"],
        "goals": [format_goal(g) for g in user_record["goals"]],
        "savings": [format_saving(s) for s in user_record["savings"]],
        "expenditures": [format_expense(e) for e in user_record["expenses"]]
    }

def format_goal(goal):
    return {
        "category": goal["category"],
        "goal_name": goal["name"],
        "goal_description": goal["description"],
        "target_amount": goal["target_amount"],
        "deadline": date.fromisoformat(goal["deadline"])
    }

def format_saving(saving):
    return {
        "savings_name": saving["name"],
        "category": saving["category"],
        "amount": saving["amount"],
        "date_created": date.fromisoformat(saving["date_created"])
    }

def format_expense(expense):
    return {
        "category": expense["category"],
        "amount": expense["amount"],
        "date": date.fromisoformat(expense["date"])
    }

# ========== SHUTDOWN HANDLER ==========
@app.on_event("shutdown")
def shutdown_event():
    db.close()

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
