"""Generate a PBKDF2 password hash for ADMIN_PASSWORD_HASH in .env."""

from getpass import getpass

from stnlp.auth.security import hash_password


def main() -> None:
    password = getpass("Nhập mật khẩu Admin (ít nhất 10 ký tự): ")
    confirmation = getpass("Nhập lại mật khẩu: ")
    if password != confirmation:
        raise SystemExit("Hai mật khẩu không trùng khớp.")
    try:
        encoded = hash_password(password)
    except ValueError as exc:
        raise SystemExit(str(exc)) from exc
    print("\nSao chép chuỗi sau vào ADMIN_PASSWORD_HASH trong file .env:")
    print(encoded)
    print("Không chia sẻ mật khẩu hoặc file .env.")


if __name__ == "__main__":
    main()
