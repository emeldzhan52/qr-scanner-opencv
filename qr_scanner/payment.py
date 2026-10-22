"""Разбор платёжных QR-кодов формата ГОСТ Р 56042-2014 («ST00012|...»).

Такие коды печатают на квитанциях и генерирует мобильное приложение
из соседнего проекта (mini-bank-app).

Пример строки:
    ST00012|Name=Иван Иванов|PersonalAcc=40817810000000000001|BankName=Мини Банк|
    BIC=044525000|CorrespAcc=30101810400000000225|Sum=150000|Purpose=Перевод
"""

from __future__ import annotations

from dataclasses import dataclass

HEADER = "ST0001"  # дальше идёт цифра кодировки: 1 - win1251, 2 - utf8, 3 - koi8-r
REQUIRED = ("Name", "PersonalAcc", "BankName", "BIC", "CorrespAcc")


@dataclass
class PaymentInfo:
    name: str
    account: str
    bank_name: str
    bic: str
    corr_account: str
    amount_rub: float | None
    purpose: str | None
    extra: dict[str, str]

    def summary(self) -> str:
        amount = f"{self.amount_rub:,.2f} ₽".replace(",", " ") if self.amount_rub else "не указана"
        lines = [
            f"Получатель: {self.name}",
            f"Счёт:       {self.account}",
            f"Банк:       {self.bank_name} (БИК {self.bic})",
            f"Сумма:      {amount}",
        ]
        if self.purpose:
            lines.append(f"Назначение: {self.purpose}")
        return "\n".join(lines)


def parse_payment(text: str) -> PaymentInfo | None:
    """Вернуть PaymentInfo или None, если строка — не платёжный QR."""
    if not text.startswith(HEADER) or "|" not in text:
        return None

    _, *pairs = text.split("|")
    fields: dict[str, str] = {}
    for pair in pairs:
        if "=" not in pair:
            continue
        key, value = pair.split("=", 1)
        fields[key.strip()] = value.strip()

    if any(key not in fields for key in REQUIRED):
        return None

    amount = None
    if "Sum" in fields:
        try:
            amount = int(fields["Sum"]) / 100  # сумма в копейках
        except ValueError:
            amount = None

    known = set(REQUIRED) | {"Sum", "Purpose"}
    return PaymentInfo(
        name=fields["Name"],
        account=fields["PersonalAcc"],
        bank_name=fields["BankName"],
        bic=fields["BIC"],
        corr_account=fields["CorrespAcc"],
        amount_rub=amount,
        purpose=fields.get("Purpose"),
        extra={k: v for k, v in fields.items() if k not in known},
    )
