"""Run an ImageNet-1k or ImageNet-21k experiment from Rawat et al. (2021)."""

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
    parser.add_argument("task", choices=("imagenet1k", "imagenet21k"))
    parser.add_argument("--method", choices=("baseline", "class", "margin"), default="class")
    parser.add_argument("--epochs", type=int, default=90)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--alpha", type=float, default=0.6)
    parser.add_argument("--rho-train", type=float, default=0.8)
    parser.add_argument("--in-domain-class-count", type=int)
    parser.add_argument("--student-width", choices=(0.35, 0.5, 0.75, 1.0, 1.25), type=float, default=0.75)
    parser.add_argument("--data-dir", default="data")
    parser.add_argument("--artifacts-dir", default="artifacts")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="auto")
    return parser.parse_args()


def main() -> None:
    args = arguments()
    if not 0 <= args.alpha <= 1 or not 0 < args.rho_train < 1:
        raise ValueError("alpha must be in [0, 1] and rho_train must be in (0, 1)")
    if args.task == "imagenet21k" and args.method == "margin":
        raise ValueError("the paper's ImageNet-21k experiment uses an oracle teacher and class distillation, not margin distillation")
    torch.manual_seed(args.seed)
    run_device = device(args.device)
    task_data = data.load(args.task, args.data_dir, args.batch_size, args.workers, args.seed)
    teacher, student = models.imagenet_models(task_data.classes, args.student_width)
    student.to(run_device)
    if teacher is not None:
        teacher.to(run_device)
        for parameter in teacher.parameters():
            parameter.requires_grad_(False)

    run_dir = Path(args.artifacts_dir) / f"{args.task}-{args.method}-{args.student_width}"
    train_teacher = evaluation_teacher = None
    if teacher is not None:
        train_teacher = teachers.cache_logits(teacher, task_data.train, run_device, run_dir / "teacher-train.pt")
        evaluation_teacher = teachers.cache_logits(teacher, task_data.evaluation, run_device, run_dir / "teacher-evaluation.pt")
    in_domain_count = args.in_domain_class_count or (300 if args.task == "imagenet1k" else 1000)
    in_domain = torch.arange(min(in_domain_count, task_data.classes))
    losses = train.train(
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
        task_data.classes,
    )
    result = evaluate.evaluate(
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
