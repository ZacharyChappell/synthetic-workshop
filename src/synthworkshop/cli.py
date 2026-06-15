"""Command-line interface for synthetic-workshop."""

from __future__ import annotations

import argparse
from collections.abc import Sequence
from pathlib import Path

from synthworkshop import __version__
from synthworkshop.datasets import catalogue_rows, get_catalogue_entry
from synthworkshop.io import inspect_export_contract
from synthworkshop.scenes.validation import validate_scene_config
from synthworkshop.workflows import render_export_gallery


def build_parser() -> argparse.ArgumentParser:
    """Build the top-level command-line parser."""

    parser = argparse.ArgumentParser(
        prog="synthworkshop",
        description=(
            "Generate, inspect, validate, and export analytic synthetic scenes."
        ),
    )
    parser.add_argument(
        "--version",
        action="version",
        version=f"%(prog)s {__version__}",
    )

    subparsers = parser.add_subparsers(dest="command", required=True)

    gui_parser = subparsers.add_parser(
        "gui",
        help="Launch the optional local scene workbench.",
    )
    gui_parser.add_argument(
        "--host",
        default="127.0.0.1",
        help="Host/interface for the Streamlit server.",
    )
    gui_parser.add_argument(
        "--port",
        type=int,
        default=8501,
        help="Port for the Streamlit server.",
    )
    gui_parser.add_argument(
        "--no-browser",
        action="store_true",
        help="Do not request browser launch.",
    )
    gui_parser.set_defaults(func=run_gui_command)

    catalogue_parser = subparsers.add_parser(
        "catalogue",
        help="List, inspect, or render curated example scenes.",
    )
    catalogue_action = catalogue_parser.add_mutually_exclusive_group(required=True)
    catalogue_action.add_argument(
        "--list",
        action="store_true",
        help="List built-in catalogue scenes.",
    )
    catalogue_action.add_argument(
        "--show",
        metavar="SCENE_ID",
        help="Show metadata for one catalogue scene.",
    )
    catalogue_action.add_argument(
        "--render",
        metavar="SCENE_ID",
        help="Render one catalogue scene.",
    )
    catalogue_parser.add_argument(
        "--output-root",
        default=None,
        help="Output directory for --render.",
    )
    catalogue_parser.add_argument(
        "--map-name",
        default=None,
        help="Scalar map to use for rendered gallery previews.",
    )
    catalogue_parser.add_argument(
        "--formats",
        nargs="+",
        default=["png"],
        help="Gallery figure formats for --render, e.g. png pdf svg.",
    )
    catalogue_parser.add_argument(
        "--dpi",
        type=int,
        default=300,
        help="DPI for raster gallery outputs.",
    )
    catalogue_parser.add_argument(
        "--with-colorbar",
        action="store_true",
        help="Include colourbars in gallery figures where supported.",
    )
    catalogue_parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Allow existing output files to be replaced.",
    )
    catalogue_parser.add_argument(
        "--no-inspect-export",
        action="store_true",
        help="Do not inspect exported files after catalogue rendering.",
    )
    catalogue_parser.add_argument(
        "--strict-export-inspection",
        action="store_true",
        help="Return a non-zero exit code for export-inspection warnings.",
    )
    catalogue_parser.set_defaults(func=run_catalogue_command)

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
            "Scalar map to use for gallery previews. If omitted, a "
            "deterministic default is selected."
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
    render_parser.add_argument(
        "--no-inspect-export",
        action="store_true",
        help="Do not inspect exported files after rendering.",
    )
    render_parser.add_argument(
        "--strict-export-inspection",
        action="store_true",
        help="Return a non-zero exit code for export-inspection warnings.",
    )
    render_parser.set_defaults(func=run_render_command)

    inspect_parser = subparsers.add_parser(
        "inspect-export",
        help="Inspect an exported scene directory.",
    )
    inspect_parser.add_argument(
        "--export-root",
        required=True,
        help="Path to an exported scene directory.",
    )
    inspect_parser.add_argument(
        "--strict",
        action="store_true",
        help="Return a non-zero exit code for warnings as well as errors.",
    )
    inspect_parser.set_defaults(func=run_inspect_export_command)

    return parser


def run_gui_command(args: argparse.Namespace) -> int:
    """Run the `synthworkshop gui` command."""

    from synthworkshop.gui import launch_gui

    return int(
        launch_gui(
            host=args.host,
            port=args.port,
            browser=not args.no_browser,
        )
    )


def run_catalogue_command(args: argparse.Namespace) -> int:
    """Run the `synthworkshop catalogue` command."""

    if args.list:
        rows = catalogue_rows()
        print("Built-in scene catalogue")
        print("======================")
        for row in rows:
            print(f"{row['scene_id']}\t{row['family']}\t{row['title']}")
        return 0

    if args.show is not None:
        entry = get_catalogue_entry(args.show)
        row = entry.to_row()
        print(f"Scene id: {row['scene_id']}")
        print(f"Title: {row['title']}")
        print(f"Family: {row['family']}")
        print(f"Config: {row['config_path']}")
        print(f"Purpose: {row['purpose']}")
        print(f"Expected appearance: {row['expected_appearance']}")
        print(f"Validation focus: {row['validation_focus']}")
        print(f"Expected failure mode: {row['expected_failure_mode']}")
        print(f"Recommended use: {row['recommended_use']}")
        print(f"Tags: {row['tags']}")
        print(f"Default output name: {row['default_output_name']}")
        if row["seed"]:
            print(f"Seed: {row['seed']}")
        if row["notes"]:
            print(f"Notes: {row['notes']}")
        return 0

    if args.render is not None:
        entry = get_catalogue_entry(args.render)
        if args.output_root is None:
            raise ValueError("--output-root is required with catalogue --render.")

        result = render_export_gallery(
            entry.config_path,
            output_root=args.output_root,
            map_name=args.map_name,
            export=True,
            gallery=True,
            formats=tuple(args.formats),
            dpi=args.dpi,
            overwrite=args.overwrite,
            with_colorbar=args.with_colorbar,
        )

        output_root = Path(args.output_root)
        print(f"Rendered catalogue scene: {entry.scene_id}")
        print(f"Selected map: {result.map_name}")
        print(f"Wrote export: {output_root / 'export'}")
        if not args.no_inspect_export:
            exit_code = _print_export_inspection(
                output_root / "export",
                strict=args.strict_export_inspection,
            )
            if exit_code != 0:
                return exit_code
        print(f"Wrote gallery: {output_root / 'gallery'}")
        return 0

    raise RuntimeError("No catalogue action was selected.")  # pragma: no cover


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
        if not args.no_inspect_export:
            exit_code = _print_export_inspection(
                output_root / "export",
                strict=args.strict_export_inspection,
            )
            if exit_code != 0:
                return exit_code
    else:
        print("Export skipped.")

    if result.gallery_written:
        print(f"Wrote gallery: {output_root / 'gallery'}")
    else:
        print("Gallery skipped.")

    return 0


def run_inspect_export_command(args: argparse.Namespace) -> int:
    """Run the `synthworkshop inspect-export` command."""

    return _print_export_inspection(args.export_root, strict=args.strict)


def _print_export_inspection(export_root: str | Path, *, strict: bool) -> int:
    """Inspect an export directory and print a compact report."""

    report = inspect_export_contract(export_root)
    counts = report.summary_counts()

    print(f"Inspected export: {report.export_root}")
    print(f"Passed: {str(report.passed).lower()}")
    print(f"Manifest entries: {len(report.manifest_rows)}")
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
    if strict and report.warnings():
        return 1
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    """Run the command-line interface."""

    parser = build_parser()
    args = parser.parse_args(argv)
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main())
