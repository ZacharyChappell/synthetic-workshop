"""Command-line interface for synthetic-workshop."""

from __future__ import annotations

import argparse
from collections.abc import Sequence
from pathlib import Path

from synthworkshop import __version__
from synthworkshop.scenes.validation import validate_scene_config
from synthworkshop.workflows import render_export_gallery


def build_parser() -> argparse.ArgumentParser:
    """Build the top-level command-line parser."""

    parser = argparse.ArgumentParser(
        prog="synthworkshop",
        description="Generate, inspect, validate and export analytic synthetic scenes.",
    )
    parser.add_argument(
        "--version",
        action="version",
        version=f"%(prog)s {__version__}",
    )

    subparsers = parser.add_subparsers(dest="command", required=True)

    validate_parser = subparsers.add_parser(
        "validate-config",
        help="Validate a YAML/JSON scene specification.",
    )
    validate_parser.add_argument(
        "--config",
        required=True,
        help="Path to a scene YAML/JSON configuration file.",
    )
    validate_parser.add_argument(
        "--render",
        action="store_true",
        help="Also render the scene to catch render-time errors.",
    )
    validate_parser.add_argument(
        "--strict",
        action="store_true",
        help="Return a non-zero exit code for warnings as well as errors.",
    )
    validate_parser.set_defaults(func=run_validate_config_command)

    render_parser = subparsers.add_parser(
        "render",
        help="Render a YAML/JSON scene specification.",
    )
    render_parser.add_argument(
        "--config",
        required=True,
        help="Path to a scene YAML/JSON configuration file.",
    )
    render_parser.add_argument(
        "--output-root",
        required=True,
        help="Directory where export and gallery outputs will be written.",
    )
    render_parser.add_argument(
        "--map-name",
        default=None,
        help=(
            "Scalar map to use for gallery previews. If omitted, a deterministic "
            "default is selected."
        ),
    )
    render_parser.add_argument(
        "--no-export",
        action="store_true",
        help="Render the scene but do not write array/table/metadata exports.",
    )
    render_parser.add_argument(
        "--no-gallery",
        action="store_true",
        help="Render the scene but do not write gallery figures.",
    )
    render_parser.add_argument(
        "--formats",
        nargs="+",
        default=["png"],
        help="Gallery figure formats to write, e.g. png pdf svg.",
    )
    render_parser.add_argument(
        "--dpi",
        type=int,
        default=300,
        help="DPI for raster gallery outputs.",
    )
    render_parser.add_argument(
        "--with-colorbar",
        action="store_true",
        help="Include colourbars in gallery figures where supported.",
    )
    render_parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Allow existing output files to be replaced.",
    )
    render_parser.set_defaults(func=run_render_command)

    return parser


def run_validate_config_command(args: argparse.Namespace) -> int:
    """Run the `synthworkshop validate-config` command."""

    report = validate_scene_config(args.config, render=args.render)
    counts = report.summary_counts()

    scene_label = report.scene_id if report.scene_id is not None else "<unknown>"
    print(f"Validated scene config: {Path(args.config)}")
    print(f"Scene id: {scene_label}")
    print(f"Rendered: {str(report.rendered).lower()}")
    print(
        "Issues: "
        f"{counts['error']} error(s), "
        f"{counts['warning']} warning(s), "
        f"{counts['info']} info message(s)"
    )

    for issue in report.issues:
        print(f"[{issue.severity}] {issue.location}: {issue.message}")

    if not report.passed:
        return 1
    if args.strict and report.warnings():
        return 1
    return 0


def run_render_command(args: argparse.Namespace) -> int:
    """Run the `synthworkshop render` command."""

    result = render_export_gallery(
        args.config,
        output_root=args.output_root,
        map_name=args.map_name,
        export=not args.no_export,
        gallery=not args.no_gallery,
        formats=tuple(args.formats),
        dpi=args.dpi,
        overwrite=args.overwrite,
        with_colorbar=args.with_colorbar,
    )

    output_root = Path(args.output_root)
    print(f"Rendered scene: {result.scene.metadata.get('scene_id', '<unknown>')}")
    print(f"Selected map: {result.map_name}")

    if result.exported:
        print(f"Wrote export: {output_root / 'export'}")
    else:
        print("Export skipped.")

    if result.gallery_written:
        print(f"Wrote gallery: {output_root / 'gallery'}")
    else:
        print("Gallery skipped.")

    return 0


def main(argv: Sequence[str] | None = None) -> int:
    """Run the command-line interface."""

    parser = build_parser()
    args = parser.parse_args(argv)
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main())
