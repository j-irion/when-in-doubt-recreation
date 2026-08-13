"""Run one public-equivalent classification experiment from the paper."""

import argparse
import json
from pathlib import Path

import torch

from . import data, evaluate, models, teachers, train


def device(name: str) -> torch.device:
    if name != "auto":
        return torch.device(name)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("task", choices=("mnli", "cifar100"))
    parser.add_argument("--method", choices=("baseline", "class", "margin"), default="margin")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--learning-rate", type=float, default=2e-5)
    parser.add_argument("--alpha", type=float, default=0.4)
    parser.add_argument("--rho-train", type=float, default=0.8)
    parser.add_argument("--in-domain-class-count", type=int, default=30)
    parser.add_argument("--cifar-depth", type=int, default=32)
    parser.add_argument("--teacher-checkpoint", default="")
    parser.add_argument("--data-dir", default="data")
    parser.add_argument("--artifacts-dir", default="artifacts")
    parser.add_argument("--device", default="auto")
    return parser.parse_args()


def main() -> None:
    args = arguments()
    if not 0 <= args.alpha <= 1 or not 0 < args.rho_train < 1:
        raise ValueError("alpha must be in [0, 1] and rho_train must be in (0, 1)")
    torch.manual_seed(0)
    run_device = device(args.device)
    task_data = data.load(args.task, args.data_dir, args.batch_size)
    teacher, student = models.load(args.task, args.teacher_checkpoint, args.cifar_depth)
    teacher.to(run_device)
    student.to(run_device)
    for parameter in teacher.parameters():
        parameter.requires_grad_(False)

    run_dir = Path(args.artifacts_dir) / f"{args.task}-{args.method}"
    train_teacher = teachers.cache_logits(
        args.task, teacher, task_data.train, run_device, run_dir / "teacher-train.pt"
    )
    evaluation_teacher = teachers.cache_logits(
        args.task, teacher, task_data.evaluation, run_device, run_dir / "teacher-evaluation.pt"
    )
    in_domain = torch.arange(min(args.in_domain_class_count, task_data.classes))
    losses = train.train(
        args.task,
        student,
        task_data.train,
        train_teacher,
        run_device,
        args.method,
        args.epochs,
        args.learning_rate,
        args.alpha,
        args.rho_train,
        in_domain,
    )
    result = evaluate.evaluate(
        args.task,
        student,
        task_data.evaluation,
        evaluation_teacher,
        run_device,
        args.method,
        in_domain,
    )
    run_dir.mkdir(parents=True, exist_ok=True)
    torch.save(student.state_dict(), run_dir / "student.pt")
    (run_dir / "metrics.json").write_text(
        json.dumps(
            {
                "task": args.task,
                "method": args.method,
                "alpha": args.alpha,
                "rho_train": args.rho_train,
                "in_domain_classes": in_domain.tolist(),
                "train_loss": losses,
                **result,
            },
            indent=2,
        )
        + "\n"
    )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
