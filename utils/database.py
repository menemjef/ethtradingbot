"""
utils/database.py — Persistence layer via SQLAlchemy.
"""

import logging
from datetime import datetime
from sqlalchemy import Column, Integer, Float, String, DateTime, Boolean, create_engine
from sqlalchemy.orm import declarative_base, sessionmaker
from config import DB_URL

logger = logging.getLogger(__name__)
Base = declarative_base()

class SignalLog(Base):
    __tablename__ = 'signals'
    id = Column(Integer, primary_key=True)
    timestamp = Column(DateTime, default=datetime.utcnow)
    symbol = Column(String)
    prediction = Column(Float)
    close_price = Column(Float)
    actual_return = Column(Float, nullable=True)
    was_correct = Column(Boolean, nullable=True)
    is_verified = Column(Boolean, default=False)

class PortfolioSnapshot(Base):
    __tablename__ = 'portfolio_history'
    id = Column(Integer, primary_key=True)
    timestamp = Column(DateTime, default=datetime.utcnow)
    balance = Column(Float)
    equity = Column(Float)
    pnl = Column(Float)

class TradeLog(Base):
    __tablename__ = 'trades'
    id = Column(Integer, primary_key=True)
    timestamp = Column(DateTime, default=datetime.utcnow)
    symbol = Column(String)
    side = Column(String)
    price = Column(Float)
    amount = Column(Float)
    status = Column(String)
    exit_price = Column(Float, nullable=True)
    tp_sl = Column(Float, nullable=True)

class Command(Base):
    __tablename__ = 'commands'
    id = Column(Integer, primary_key=True)
    timestamp = Column(DateTime, default=datetime.utcnow)
    command = Column(String)     # 'CLOSE_ALL', 'PAUSE', 'RESUME'
    status = Column(String, default='PENDING') # 'PENDING', 'EXECUTED'

# Database Engine and Session - Lazy initialization
_engine = None
_SessionLocal = None

def _get_engine():
    """Lazy initialization of database engine."""
    global _engine
    if _engine is None:
        _engine = create_engine(DB_URL)
    return _engine

def _get_session():
    """Get a new database session."""
    global _SessionLocal
    if _SessionLocal is None:
        _SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=_get_engine())
    return _SessionLocal()

def SessionLocal():
    """Public alias to get a new database session."""
    return _get_session()

def init_db():
    """Create tables if they don't exist."""
    try:
        Base.metadata.create_all(bind=_get_engine())
        logger.info("Database initialized successfully.")
    except Exception as e:
        logger.error(f"Failed to initialize database: {e}")

def save_signal(symbol: str, prediction: float, close_price: float):
    session = _get_session()
    try:
        new_signal = SignalLog(
            symbol=symbol,
            prediction=prediction,
            close_price=close_price
        )
        session.add(new_signal)
        session.commit()
    except Exception as e:
        logger.error(f"Failed to save signal to DB: {e}")
        session.rollback()
    finally:
        session.close()

def save_trade(symbol, side, price, amount, status, exit_price=None, tp_sl=None):
    session = _get_session()
    try:
        new_trade = TradeLog(
            symbol=symbol,
            side=side,
            price=price,
            amount=amount,
            status=status,
            exit_price=exit_price,
            tp_sl=tp_sl
        )
        session.add(new_trade)
        session.commit()
    except Exception as e:
        logger.error(f"Failed to save trade to DB: {e}")
        session.rollback()
    finally:
        session.close()

def save_portfolio_snapshot(balance, equity, pnl):
    session = _get_session()
    try:
        snap = PortfolioSnapshot(balance=balance, equity=equity, pnl=pnl)
        session.add(snap)
        session.commit()
    except Exception as e:
        logger.error(f"Failed to save portfolio snapshot: {e}")
        session.rollback()
    finally:
        session.close()

def save_command(cmd_text: str):
    session = _get_session()
    try:
        new_cmd = Command(command=cmd_text, status='PENDING')
        session.add(new_cmd)
        session.commit()
        logger.info(f"Command '{cmd_text}' saved to DB.")
    except Exception as e:
        logger.error(f"Failed to save command to DB: {e}")
        session.rollback()
    finally:
        session.close()

def get_pending_commands():
    session = _get_session()
    try:
        cmds = session.query(Command).filter(Command.status == 'PENDING').all()
        return cmds
    except Exception as e:
        logger.error(f"Failed to fetch pending commands: {e}")
        return []
    finally:
        session.close()

def mark_command_executed(cmd_id):
    session = _get_session()
    try:
        cmd = session.query(Command).filter(Command.id == cmd_id).first()
        if cmd:
            cmd.status = 'EXECUTED'
            session.commit()
    except Exception as e:
        logger.error(f"Failed to mark command executed: {e}")
        session.rollback()
    finally:
        session.close()
