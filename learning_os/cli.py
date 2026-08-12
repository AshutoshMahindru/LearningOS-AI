import argparse


def main():
    parser = argparse.ArgumentParser(description="Learning OS CLI")
    parser.add_argument("command", choices=["start", "status", "next", "gate", "review"])
    parser.add_argument("mission", nargs="?")
    args = parser.parse_args()

    if args.command == "start":
        print(f"Starting mission {args.mission}")
    elif args.command == "status":
        print("Learner status")
    elif args.command == "next":
        print("Next recommended action")
    elif args.command == "gate":
        print("Competency gate")
    elif args.command == "review":
        print("Review mode")


if __name__ == "__main__":
    main()
