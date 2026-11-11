"""SQLAlchemy ORM models for users, reports, predictions and moderation audit trail."""

from __future__ import annotations

from datetime import datetime, timezone

from sqlalchemy import Boolean, DateTime, Float, ForeignKey, Integer, JSON, String, Text
from sqlalchemy.orm import Mapped, mapped_column, relationship

from stnlp.database.base import Base


def utc_now():
    return datetime.now(timezone.utc)


class Category(Base):
    __tablename__ = "categories"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    code: Mapped[str] = mapped_column(String(3), unique=True, index=True)
    name: Mapped[str] = mapped_column(String(160), unique=True)
    description: Mapped[str | None] = mapped_column(Text, nullable=True)

    reports: Mapped[list["Report"]] = relationship(
        back_populates="confirmed_category", foreign_keys="Report.confirmed_category_id"
    )
    predictions: Mapped[list["Prediction"]] = relationship(back_populates="category")


class AdminUser(Base):
    __tablename__ = "admins"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    username: Mapped[str] = mapped_column(String(80), unique=True, index=True)
    password_hash: Mapped[str] = mapped_column(String(255))
    role: Mapped[str] = mapped_column(String(32), default="admin")
    is_active: Mapped[bool] = mapped_column(Boolean, default=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utc_now)


class Report(Base):
    __tablename__ = "reports"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    report_code: Mapped[str] = mapped_column(String(32), unique=True, index=True)
    text: Mapped[str] = mapped_column(Text)
    latitude: Mapped[float | None] = mapped_column(Float, nullable=True)
    longitude: Mapped[float | None] = mapped_column(Float, nullable=True)
    image_url: Mapped[str | None] = mapped_column(String(1000), nullable=True)
    status: Mapped[str] = mapped_column(String(32), default="new", index=True)
    confirmed_category_id: Mapped[int | None] = mapped_column(
        ForeignKey("categories.id"), nullable=True
    )
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utc_now)

    confirmed_category: Mapped[Category | None] = relationship(
        back_populates="reports", foreign_keys=[confirmed_category_id]
    )
    predictions: Mapped[list["Prediction"]] = relationship(
        back_populates="report", cascade="all, delete-orphan"
    )
    moderation_logs: Mapped[list["ModerationLog"]] = relationship(
        back_populates="report", cascade="all, delete-orphan"
    )


class Prediction(Base):
    __tablename__ = "predictions"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    report_id: Mapped[int] = mapped_column(ForeignKey("reports.id"), index=True)
    category_id: Mapped[int] = mapped_column(ForeignKey("categories.id"))
    confidence: Mapped[float] = mapped_column(Float)
    probabilities: Mapped[dict | None] = mapped_column(JSON, nullable=True)
    model_name: Mapped[str] = mapped_column(String(64))
    model_version: Mapped[str] = mapped_column(String(64))
    needs_review: Mapped[bool] = mapped_column(Boolean, default=False, index=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utc_now)

    report: Mapped[Report] = relationship(back_populates="predictions")
    category: Mapped[Category] = relationship(back_populates="predictions")


class ModerationLog(Base):
    __tablename__ = "moderation_logs"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    report_id: Mapped[int] = mapped_column(ForeignKey("reports.id"), index=True)
    admin_identifier: Mapped[str] = mapped_column(String(120), default="admin")
    old_category_id: Mapped[int | None] = mapped_column(ForeignKey("categories.id"), nullable=True)
    new_category_id: Mapped[int] = mapped_column(ForeignKey("categories.id"))
    note: Mapped[str | None] = mapped_column(Text, nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utc_now)

    report: Mapped[Report] = relationship(back_populates="moderation_logs")
