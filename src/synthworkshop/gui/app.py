"""Streamlit scene workbench for synthetic-workshop."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from synthworkshop.datasets import get_catalogue_entry
from synthworkshop.gui.catalogue_package import (
    CATALOGUE_FAMILIES,
    default_catalogue_package_dir,
    export_catalogue_package,
    humanise_scene_id,
    read_catalogue_package,
    scene_id_from_text,
)
from synthworkshop.gui.file_io import (
    decode_uploaded_scene_bytes,
    default_saved_scene_path,
    read_scene_text_file,
    save_scene_text_file,
)
from synthworkshop.gui.placement import (
    geometry_controls_for_object,
    grid_extent_mm,
    update_object_geometry,
)
from synthworkshop.gui.profiles import (
    PROFILE_KINDS,
    apply_profile_updates,
    profile_controls_for_profile,
    profile_for_object,
    profile_template,
    replace_object_profile,
)
from synthworkshop.gui.scene_settings import (
    LABEL_MODES,
    OVERLAP_POLICIES,
    SCALAR_BLEND_MODES,
    format_numeric_list,
    format_string_list,
    scene_settings_from_text,
    update_scene_settings,
)
from synthworkshop.gui.state import (
    catalogue_rows,
    catalogue_scene_ids,
    default_output_root,
    gallery_png_paths,
    read_scene_text,
    render_catalogue_scene,
    render_preview_scene_config_text,
    render_scene_config_text,
    validate_catalogue_scene,
    validate_scene_config_text,
)
from synthworkshop.gui.summary import (
    build_scene_summary,
    default_summary_path,
    save_scene_summary_json,
    summary_to_json,
)
from synthworkshop.gui.yaml_editor import (
    add_object_to_scene_text,
    apply_field_edits,
    delete_object_from_scene_text,
    duplicate_object_in_scene_text,
    duplicate_scene_text,
    flatten_editable_fields,
    format_edit_value,
    get_object,
    make_ellipsoid_object,
    make_minimal_tube_scene_text,
    make_sphere_object,
    make_tube_object,
    object_ids,
    object_summary_rows,
    replace_object,
    scene_grid_centre,
    suggest_next_label,
    suggest_object_id,
)


def _import_streamlit() -> Any:
    """Import Streamlit lazily with a helpful error."""

    try:
        import streamlit as st
    except ImportError as exc:  # pragma: no cover - optional dependency guard
        raise RuntimeError(
            "The GUI requires Streamlit. Install it with "
            "`python -m pip install -e '.[gui]'`."
        ) from exc
    return st


def main() -> None:
    """Run the Streamlit GUI."""

    st = _import_streamlit()

    st.set_page_config(
        page_title="synthetic-workshop",
        page_icon="🧪",
        layout="wide",
    )

    st.title("synthetic-workshop scene workbench")
    st.caption("Build, validate, render, inspect and export analytic synthetic scenes.")

    scene_ids = catalogue_scene_ids()
    if not scene_ids:
        st.error("No catalogue scenes are available.")
        return

    with st.sidebar:
        st.header("Scene")
        selected_scene_id = st.selectbox(
            "Catalogue scene",
            scene_ids,
            index=0,
        )

        entry = get_catalogue_entry(selected_scene_id)
        st.write(f"**Family:** {entry.family}")
        st.write(f"**Config:** `{entry.config_path}`")

        output_root_text = st.text_input(
            "Output root",
            value=str(default_output_root(selected_scene_id)),
        )

        st.header("Render options")
        map_name_text = st.text_input(
            "Map name",
            value="",
            help="Leave blank to select the default scalar map.",
        )
        formats = st.multiselect(
            "Gallery formats",
            options=["png", "pdf", "svg"],
            default=["png"],
        )
        dpi = st.number_input(
            "DPI",
            min_value=72,
            max_value=1200,
            value=200,
            step=50,
        )
        with_colorbar = st.checkbox("Include colourbars", value=False)
        overwrite = st.checkbox("Overwrite outputs", value=True)
        render_edited_yaml = st.checkbox(
            "Use edited YAML",
            value=True,
            help=(
                "When enabled, validation/rendering uses the current YAML editor "
                "text rather than the catalogue file on disk."
            ),
        )

        if st.button("Reset YAML from catalogue"):
            state_key = f"scene_text::{selected_scene_id}"
            st.session_state[state_key] = read_scene_text(entry)
            st.rerun()

    output_root = Path(output_root_text)
    map_name = map_name_text.strip() or None
    safe_formats = tuple(formats) if formats else ("png",)

    state_key = f"scene_text::{selected_scene_id}"
    if state_key not in st.session_state:
        st.session_state[state_key] = read_scene_text(entry)

    (
        tab_overview,
        tab_file,
        tab_package,
        tab_summary,
        tab_scene_settings,
        tab_new_scene,
        tab_profile,
        tab_placement,
        tab_objects,
        tab_yaml,
        tab_validate,
        tab_render,
        tab_catalogue,
    ) = st.tabs(
        [
            "Overview",
            "File",
            "Package",
            "Summary",
            "Scene settings",
            "New scene",
            "Profile",
            "Placement",
            "Object editor",
            "YAML",
            "Validate",
            "Render",
            "Catalogue",
        ]
    )

    with tab_overview:
        st.subheader(entry.title)
        st.write(entry.purpose)

        col_a, col_b = st.columns(2)
        with col_a:
            st.markdown("**Expected appearance**")
            st.write(entry.expected_appearance)
        with col_b:
            st.markdown("**Validation focus**")
            if entry.validation_focus:
                st.write("\n".join(f"- {item}" for item in entry.validation_focus))
            else:
                st.write("No validation-focus metadata recorded.")

        if entry.notes:
            st.markdown("**Notes**")
            st.write(entry.notes)

    with tab_file:
        st.subheader("Load and save scene specifications")
        current_text = str(st.session_state[state_key])

        st.markdown("### Upload from browser")
        uploaded_file = st.file_uploader(
            "Upload YAML/JSON scene config",
            type=["yml", "yaml", "json"],
            accept_multiple_files=False,
        )
        if uploaded_file is not None:
            st.write(f"Uploaded: `{uploaded_file.name}`")
            if st.button("Load uploaded file into editor", type="primary"):
                try:
                    uploaded_text = decode_uploaded_scene_bytes(
                        uploaded_file.getvalue()
                    )
                    st.session_state[state_key] = uploaded_text
                    st.session_state[f"loaded_source::{selected_scene_id}"] = (
                        uploaded_file.name
                    )
                except Exception as exc:
                    st.error(f"{type(exc).__name__}: {exc}")
                else:
                    st.success("Loaded uploaded scene into the YAML editor.")
                    st.rerun()

        st.markdown("### Open local file")
        with st.form(f"open_scene_file::{selected_scene_id}"):
            open_path = st.text_input(
                "Local scene path",
                value="",
                placeholder="examples/basic_tube.yml",
                help=(
                    "Path on the machine running Streamlit, not necessarily "
                    "your browser device."
                ),
            )
            open_submitted = st.form_submit_button("Open local file")

        if open_submitted:
            try:
                opened_text = read_scene_text_file(open_path)
                st.session_state[state_key] = opened_text
                st.session_state[f"loaded_source::{selected_scene_id}"] = open_path
            except Exception as exc:
                st.error(f"{type(exc).__name__}: {exc}")
            else:
                st.success(f"Loaded `{open_path}` into the YAML editor.")
                st.rerun()

        st.markdown("### Save current editor text")
        default_save_path = default_saved_scene_path(
            output_root=output_root,
            scene_id=selected_scene_id,
        )
        with st.form(f"save_scene_file::{selected_scene_id}"):
            save_path = st.text_input(
                "Save path",
                value=str(default_save_path),
                help="Path on the machine running Streamlit.",
            )
            save_overwrite = st.checkbox(
                "Overwrite existing file",
                value=False,
            )
            save_submitted = st.form_submit_button("Save YAML to disk")

        if save_submitted:
            try:
                written_path = save_scene_text_file(
                    current_text,
                    save_path,
                    overwrite=save_overwrite,
                )
            except Exception as exc:
                st.error(f"{type(exc).__name__}: {exc}")
            else:
                st.success(f"Saved scene YAML to `{written_path}`.")

        st.markdown("### Download current editor text")
        st.download_button(
            "Download YAML",
            data=current_text,
            file_name=f"{selected_scene_id}.yml",
            mime="text/yaml",
        )

        loaded_source = st.session_state.get(
            f"loaded_source::{selected_scene_id}",
            None,
        )
        if loaded_source:
            st.info(f"Current editor source: `{loaded_source}`")

    with tab_package:
        st.subheader("Catalogue package export/import")
        current_text = str(st.session_state[state_key])

        try:
            current_scene_id = scene_id_from_text(current_text) or selected_scene_id
        except Exception:
            current_scene_id = selected_scene_id

        st.markdown("### Export current scene as a catalogue package")

        default_package_dir = default_catalogue_package_dir(
            output_root=output_root,
            scene_id=current_scene_id,
        )

        with st.form(f"export_catalogue_package::{selected_scene_id}"):
            package_dir = st.text_input(
                "Package directory",
                value=str(default_package_dir),
                help=(
                    "Directory on the machine running Streamlit. The GUI will "
                    "write scene.yml, README.md, and metadata JSON files here."
                ),
            )
            package_title = st.text_input(
                "Title",
                value=humanise_scene_id(current_scene_id),
            )
            package_family = st.selectbox(
                "Family",
                CATALOGUE_FAMILIES,
                index=0,
            )
            package_purpose = st.text_area(
                "Purpose",
                value="",
                height=90,
                placeholder=("What validation problem does this scene test?"),
            )
            package_expected = st.text_area(
                "Expected appearance",
                value="",
                height=90,
                placeholder=("What should the rendered scene look like?"),
            )
            package_focus = st.text_input(
                "Validation focus",
                value="",
                placeholder="tube rendering, overlap reporting, profile recovery",
                help="Comma-separated list.",
            )
            package_notes = st.text_area(
                "Notes",
                value="",
                height=80,
            )
            package_render_summary = st.checkbox(
                "Render scene while building summary",
                value=False,
            )
            package_overwrite = st.checkbox(
                "Overwrite existing package directory",
                value=False,
            )

            export_package = st.form_submit_button(
                "Export catalogue package",
                type="primary",
            )

        if export_package:
            try:
                package = export_catalogue_package(
                    text=current_text,
                    package_dir=package_dir,
                    title=package_title,
                    family=package_family,
                    purpose=package_purpose,
                    expected_appearance=package_expected,
                    validation_focus=package_focus,
                    notes=package_notes,
                    overwrite=package_overwrite,
                    render_summary=package_render_summary,
                )
            except Exception as exc:
                st.error(f"{type(exc).__name__}: {exc}")
            else:
                st.success(f"Exported catalogue package to `{package.package_dir}`.")
                st.dataframe(package.path_rows(), use_container_width=True)
                st.json(package.entry, expanded=False)

        st.markdown("### Import catalogue package")
        with st.form(f"import_catalogue_package::{selected_scene_id}"):
            import_package_dir = st.text_input(
                "Package directory to import",
                value="",
                placeholder=str(default_package_dir),
            )
            import_package = st.form_submit_button("Load package scene into editor")

        if import_package:
            try:
                package = read_catalogue_package(import_package_dir)
                st.session_state[state_key] = package.scene_text
                st.session_state[f"loaded_source::{selected_scene_id}"] = str(
                    package.scene_path
                )
            except Exception as exc:
                st.error(f"{type(exc).__name__}: {exc}")
            else:
                st.success(
                    f"Loaded package scene `{package.entry.get('scene_id', '')}` "
                    "into the YAML editor."
                )
                st.json(package.entry, expanded=False)
                st.rerun()

    with tab_summary:
        st.subheader("Scene summary and metadata")
        current_text = str(st.session_state[state_key])

        render_for_summary = st.checkbox(
            "Render scene for array-level summary",
            value=False,
            help=(
                "When enabled, the summary also checks that the scene can render "
                "and records rendered scalar maps, label-map shape, and object masks."
            ),
        )

        try:
            summary = build_scene_summary(
                current_text,
                render=render_for_summary,
            )
        except Exception as exc:
            st.error(f"{type(exc).__name__}: {exc}")
            summary = None

        if summary is not None:
            scene_block = summary["scene"]
            grid_block = summary["grid"]
            composition_block = summary["composition"]
            mask_block = summary["mask_rules"]
            objects_block = summary["objects"]
            render_block = summary["render"]

            st.markdown("### Scene")
            col_scene_a, col_scene_b, col_scene_c = st.columns(3)
            col_scene_a.metric("Scene ID", scene_block["id"] or "<unset>")
            col_scene_b.metric("Schema", summary["schema_version"] or "<unset>")
            col_scene_c.metric("Objects", objects_block["n_objects"])

            if scene_block["description"]:
                st.write(scene_block["description"])

            st.markdown("### Grid and composition")
            col_grid, col_comp, col_masks = st.columns(3)
            with col_grid:
                st.write("**Grid**")
                st.json(grid_block, expanded=False)
            with col_comp:
                st.write("**Composition**")
                st.json(composition_block, expanded=False)
            with col_masks:
                st.write("**Mask rules**")
                st.json(mask_block, expanded=False)

            st.markdown("### Objects")
            metric_cols = st.columns(4)
            metric_cols[0].metric(
                "Valid object mappings",
                objects_block["n_valid_object_mappings"],
            )
            metric_cols[1].metric(
                "Scalar maps",
                len(objects_block["map_names"]),
            )
            metric_cols[2].metric(
                "Duplicate IDs",
                len(objects_block["duplicate_ids"]),
            )
            metric_cols[3].metric(
                "Duplicate labels",
                len(objects_block["duplicate_labels"]),
            )

            st.write("**Object table**")
            st.dataframe(objects_block["rows"], use_container_width=True)

            col_counts_a, col_counts_b, col_counts_c = st.columns(3)
            with col_counts_a:
                st.write("**By kind**")
                st.json(objects_block["kind_counts"], expanded=False)
            with col_counts_b:
                st.write("**By role**")
                st.json(objects_block["role_counts"], expanded=False)
            with col_counts_c:
                st.write("**By scalar map**")
                st.json(objects_block["map_counts"], expanded=False)

            if objects_block["duplicate_ids"] or objects_block["duplicate_labels"]:
                st.warning(
                    "The current scene contains duplicate object IDs or labels. "
                    "Validation/rendering may fail or produce ambiguous labels."
                )

            st.markdown("### Render summary")
            if render_block["attempted"]:
                if render_block["passed"]:
                    st.success("Render summary passed.")
                else:
                    st.error(render_block["error"])
                st.json(render_block, expanded=False)
            else:
                st.info("Enable render summary above to inspect rendered arrays.")

            st.markdown("### Export summary JSON")
            summary_json = summary_to_json(summary)
            summfile = f"{scene_block['id'] or selected_scene_id}_scene_summary.json"
            st.download_button(
                "Download scene_summary.json",
                data=summary_json + "\n",
                file_name=summfile,
                mime="application/json",
            )

            default_path = default_summary_path(
                output_root=output_root,
                scene_id=scene_block["id"] or selected_scene_id,
            )
            with st.form(f"save_summary::{selected_scene_id}"):
                summary_path = st.text_input(
                    "Save summary path",
                    value=str(default_path),
                )
                summary_overwrite = st.checkbox(
                    "Overwrite existing summary",
                    value=True,
                )
                save_summary = st.form_submit_button("Save summary JSON to disk")

            if save_summary:
                try:
                    written_summary = save_scene_summary_json(
                        summary,
                        summary_path,
                        overwrite=summary_overwrite,
                    )
                except Exception as exc:
                    st.error(f"{type(exc).__name__}: {exc}")
                else:
                    st.success(f"Saved scene summary to `{written_summary}`.")

    with tab_scene_settings:
        st.subheader("Scene settings")
        scene_text = str(st.session_state[state_key])

        try:
            settings = scene_settings_from_text(scene_text)
        except Exception as exc:
            st.error(f"{type(exc).__name__}: {exc}")
            settings = None

        if settings is not None:
            with st.form(f"scene_settings::{selected_scene_id}"):
                st.markdown("### Metadata")
                settings_scene_id = st.text_input(
                    "Scene ID",
                    value=settings["scene_id"],
                )
                settings_description = st.text_area(
                    "Description",
                    value=settings["description"],
                    height=90,
                )

                st.markdown("### Grid")
                settings_shape = st.text_input(
                    "Shape",
                    value=format_numeric_list(settings["shape"]),
                    help="Two or three positive integers, e.g. [32, 32, 32].",
                )
                settings_spacing = st.text_input(
                    "Spacing",
                    value=format_numeric_list(settings["spacing"]),
                    help="Two or three positive values, e.g. [1.0, 1.0, 1.0].",
                )
                settings_origin = st.text_input(
                    "Origin",
                    value=format_numeric_list(settings["origin"]),
                    help="Optional. Leave empty to omit origin.",
                )
                settings_axis_names = st.text_input(
                    "Axis names",
                    value=format_string_list(settings["axis_names"]),
                    help="Optional. Example: [i, j, k].",
                )

                st.markdown("### Composition")
                settings_label_mode = st.selectbox(
                    "Label mode",
                    LABEL_MODES,
                    index=LABEL_MODES.index(settings["label_mode"])
                    if settings["label_mode"] in LABEL_MODES
                    else 0,
                )
                settings_scalar_blend = st.selectbox(
                    "Scalar blend",
                    SCALAR_BLEND_MODES,
                    index=SCALAR_BLEND_MODES.index(settings["scalar_blend"])
                    if settings["scalar_blend"] in SCALAR_BLEND_MODES
                    else 0,
                )
                settings_overlap_policy = st.selectbox(
                    "Overlap policy",
                    OVERLAP_POLICIES,
                    index=OVERLAP_POLICIES.index(settings["overlap_policy"])
                    if settings["overlap_policy"] in OVERLAP_POLICIES
                    else 1,
                )

                st.markdown("### Mask roles")
                settings_target_roles = st.text_input(
                    "Target roles",
                    value=format_string_list(settings["target_roles"]),
                    help="YAML list or comma-separated role names.",
                )
                settings_analysis_roles = st.text_input(
                    "Analysis roles",
                    value=format_string_list(settings["analysis_roles"]),
                    help="YAML list or comma-separated role names.",
                )

                apply_settings = st.form_submit_button(
                    "Apply scene settings",
                    type="primary",
                )

            if apply_settings:
                try:
                    st.session_state[state_key] = update_scene_settings(
                        scene_text,
                        scene_id=settings_scene_id,
                        description=settings_description,
                        shape=settings_shape,
                        spacing=settings_spacing,
                        origin=settings_origin,
                        axis_names=settings_axis_names,
                        label_mode=settings_label_mode,
                        scalar_blend=settings_scalar_blend,
                        overlap_policy=settings_overlap_policy,
                        target_roles=settings_target_roles,
                        analysis_roles=settings_analysis_roles,
                        schema_version=settings["schema_version"],
                    )
                except Exception as exc:
                    st.error(f"{type(exc).__name__}: {exc}")
                else:
                    st.success("Updated scene-level settings in the YAML editor.")
                    st.rerun()

    with tab_new_scene:
        st.subheader("Create or extend a scene")

        st.markdown("### Create a minimal tube scene")
        with st.form(f"new_scene::{selected_scene_id}"):
            new_scene_id = st.text_input(
                "New scene ID",
                value="new_tube_scene",
            )
            new_description = st.text_input(
                "Description",
                value="New straight circular tube scene.",
            )
            new_shape = st.text_input(
                "Grid shape",
                value="[32, 32, 32]",
            )
            new_spacing = st.text_input(
                "Grid spacing",
                value="[1.0, 1.0, 1.0]",
            )
            new_map_name = st.text_input(
                "Map name",
                value="fa_like",
            )
            new_radius = st.number_input(
                "Tube radius (mm)",
                min_value=0.1,
                value=3.0,
                step=0.5,
            )

            create_scene = st.form_submit_button("Create minimal tube scene")

        if create_scene:
            try:
                import yaml

                shape = yaml.safe_load(new_shape)
                spacing = yaml.safe_load(new_spacing)
                st.session_state[state_key] = make_minimal_tube_scene_text(
                    scene_id=new_scene_id,
                    description=new_description,
                    shape=shape,
                    spacing=spacing,
                    map_name=new_map_name,
                    radius_mm=float(new_radius),
                )
            except Exception as exc:
                st.error(f"{type(exc).__name__}: {exc}")
            else:
                st.success(f"Created scene `{new_scene_id}` in the YAML editor.")
                st.rerun()

        st.markdown("### Duplicate current scene")
        with st.form(f"duplicate_scene::{selected_scene_id}"):
            duplicate_id = st.text_input(
                "Duplicate scene ID",
                value=f"{selected_scene_id}_copy",
            )
            duplicate_suffix = st.text_input(
                "Description suffix",
                value="Duplicated and edited in the GUI.",
            )
            duplicate_scene = st.form_submit_button("Duplicate current YAML")

        if duplicate_scene:
            try:
                st.session_state[state_key] = duplicate_scene_text(
                    str(st.session_state[state_key]),
                    new_scene_id=duplicate_id,
                    description_suffix=duplicate_suffix,
                )
            except Exception as exc:
                st.error(f"{type(exc).__name__}: {exc}")
            else:
                st.success(f"Duplicated current scene as `{duplicate_id}`.")
                st.rerun()

        st.markdown("### Add object to current scene")
        scene_text = str(st.session_state[state_key])

        try:
            centre = scene_grid_centre(scene_text)
            next_label = suggest_next_label(scene_text)
            tube_id = suggest_object_id(scene_text, "new_tube")
            sphere_id = suggest_object_id(scene_text, "sphere_inclusion")
            ellipsoid_id = suggest_object_id(scene_text, "ellipsoid_environment")
        except Exception:
            centre = [16.0, 16.0, 16.0]
            next_label = 2
            tube_id = "new_tube"
            sphere_id = "sphere_inclusion"
            ellipsoid_id = "ellipsoid_environment"

        object_kind = st.selectbox(
            "Object preset",
            ["tube", "sphere", "ellipsoid"],
            index=0,
        )

        with st.form(f"add_object::{selected_scene_id}::{object_kind}"):
            if object_kind == "tube":
                add_id = st.text_input("Object ID", value=tube_id)
            elif object_kind == "sphere":
                add_id = st.text_input("Object ID", value=sphere_id)
            else:
                add_id = st.text_input("Object ID", value=ellipsoid_id)

            add_role = st.selectbox(
                "Role",
                [
                    "target",
                    "analysis_support",
                    "environment",
                    "distractor",
                    "inclusion",
                    "background",
                ],
                index=2 if object_kind != "tube" else 1,
            )
            add_label = st.number_input(
                "Label",
                min_value=1,
                value=int(next_label),
                step=1,
            )
            add_priority = st.number_input(
                "Priority",
                value=1,
                step=1,
            )
            add_map_name = st.text_input(
                "Map name",
                value="fa_like" if object_kind == "tube" else "wm_pve_like",
            )

            if object_kind == "tube":
                start_mm = st.text_input(
                    "Start mm",
                    value=f"[{centre[0] - 8.0:.1f},{centre[1] + 6.0:.1f}"
                    ",{centre[2]:.1f}]",
                )
                end_mm = st.text_input(
                    "End mm",
                    value=f"[{centre[0] + 8.0:.1f},{centre[1] + 6.0:.1f}"
                    ",{centre[2]:.1f}]",
                )
                radius_mm = st.number_input(
                    "Radius mm",
                    min_value=0.1,
                    value=2.0,
                    step=0.5,
                )
                profile_kind = st.selectbox(
                    "Profile",
                    ["constant", "linear_radial", "gaussian_radial", "edge_enhanced"],
                    index=0,
                )

            elif object_kind == "sphere":
                centre_mm = st.text_input(
                    "Centre mm",
                    value=f"[{centre[0]:.1f}, {centre[1] + 5.0:.1f}, {centre[2]:.1f}]",
                )
                radius_mm = st.number_input(
                    "Radius mm",
                    min_value=0.1,
                    value=2.5,
                    step=0.5,
                )
                value = st.number_input(
                    "Scalar value",
                    value=1.25,
                    step=0.25,
                )

            else:
                centre_mm = st.text_input(
                    "Centre mm",
                    value=f"[{centre[0]:.1f}, {centre[1] + 6.0:.1f}, {centre[2]:.1f}]",
                )
                radii_mm = st.text_input(
                    "Radii mm",
                    value="[4.0, 2.0, 3.0]",
                )
                value = st.number_input(
                    "Scalar value",
                    value=0.35,
                    step=0.05,
                )

            add_object = st.form_submit_button("Add object")

        if add_object:
            try:
                import yaml

                if object_kind == "tube":
                    obj = make_tube_object(
                        object_id=add_id,
                        role=add_role,
                        label=int(add_label),
                        priority=int(add_priority),
                        map_name=add_map_name,
                        start_mm=yaml.safe_load(start_mm),
                        end_mm=yaml.safe_load(end_mm),
                        radius_mm=float(radius_mm),
                        profile_kind=profile_kind,
                    )
                elif object_kind == "sphere":
                    obj = make_sphere_object(
                        object_id=add_id,
                        role=add_role,
                        label=int(add_label),
                        priority=int(add_priority),
                        map_name=add_map_name,
                        centre_mm=yaml.safe_load(centre_mm),
                        radius_mm=float(radius_mm),
                        value=float(value),
                    )
                else:
                    obj = make_ellipsoid_object(
                        object_id=add_id,
                        role=add_role,
                        label=int(add_label),
                        priority=int(add_priority),
                        map_name=add_map_name,
                        centre_mm=yaml.safe_load(centre_mm),
                        radii_mm=yaml.safe_load(radii_mm),
                        value=float(value),
                    )

                st.session_state[state_key] = add_object_to_scene_text(
                    str(st.session_state[state_key]),
                    obj,
                )
            except Exception as exc:
                st.error(f"{type(exc).__name__}: {exc}")
            else:
                st.success(f"Added `{add_id}` to the current YAML scene.")
                st.rerun()

    with tab_profile:
        st.subheader("Scalar profile editor")
        scene_text = str(st.session_state[state_key])

        try:
            ids = object_ids(scene_text)
        except Exception as exc:
            st.error(f"{type(exc).__name__}: {exc}")
            ids = []

        if not ids:
            st.info("No editable objects were found in this scene.")
        else:
            selected_profile_object_id = st.selectbox(
                "Object profile",
                ids,
                index=0,
                key=f"profile_object::{selected_scene_id}",
            )

            try:
                selected_profile_object = get_object(
                    scene_text,
                    selected_profile_object_id,
                )
                current_profile = profile_for_object(selected_profile_object)
            except Exception as exc:
                st.error(f"{type(exc).__name__}: {exc}")
                current_profile = {}

            if current_profile:
                current_kind = str(current_profile.get("kind", "linear_radial"))
                default_index = (
                    PROFILE_KINDS.index(current_kind)
                    if current_kind in PROFILE_KINDS
                    else 1
                )

                selected_profile_kind = st.selectbox(
                    "Profile kind",
                    PROFILE_KINDS,
                    index=default_index,
                    key=f"profile_kind::{selected_scene_id}::{selected_profile_object_id}",
                )

                working_profile = profile_template(
                    selected_profile_kind,
                    existing=current_profile,
                )
                controls = profile_controls_for_profile(working_profile)

                st.caption(
                    "Profile edits update the YAML profile block directly. "
                    "Use Apply and preview for a lightweight gallery-only render."
                )

                updates: dict[str, float] = {}
                with st.form(
                    f"profile_editor::{selected_scene_id}::{selected_profile_object_id}"
                ):
                    for control in controls:
                        if control.help:
                            st.caption(control.help)

                        updates[control.key] = st.number_input(
                            control.label,
                            min_value=control.min_value,
                            max_value=control.max_value,
                            value=float(control.value),
                            step=control.step,
                            key=(
                                f"{selected_scene_id}::"
                                f"{selected_profile_object_id}::"
                                f"profile::{control.key}"
                            ),
                        )

                    submitted = st.form_submit_button("Apply profile changes")
                    preview_submitted = st.form_submit_button(
                        "Apply and preview",
                        type="primary",
                    )

                if submitted or preview_submitted:
                    try:
                        updated_profile = profile_template(
                            selected_profile_kind,
                            existing=current_profile,
                        )
                        updated_profile = apply_profile_updates(
                            updated_profile,
                            updates,
                        )
                        updated_text = replace_object_profile(
                            scene_text,
                            selected_profile_object_id,
                            updated_profile,
                        )
                        st.session_state[state_key] = updated_text
                    except Exception as exc:
                        st.error(f"{type(exc).__name__}: {exc}")
                    else:
                        st.success(
                            f"Updated profile for `{selected_profile_object_id}`."
                        )

                        if preview_submitted:
                            preview_root = output_root / "profile_preview"
                            try:
                                preview_result = render_preview_scene_config_text(
                                    scene_id=selected_scene_id,
                                    text=updated_text,
                                    output_root=output_root,
                                    map_name=map_name,
                                    formats=("png",),
                                    dpi=160,
                                    overwrite=True,
                                    with_colorbar=with_colorbar,
                                    preview_name="profile_preview",
                                )
                            except Exception as exc:
                                st.error(f"{type(exc).__name__}: {exc}")
                            else:
                                st.info(
                                    "Preview rendered using map "
                                    f"`{preview_result.map_name}`."
                                )
                                preview_images = gallery_png_paths(preview_root)
                                if preview_images:
                                    st.markdown("### Profile preview")
                                    for image_path in preview_images:
                                        st.image(
                                            str(image_path),
                                            caption=str(image_path),
                                            use_container_width=True,
                                        )
                                else:
                                    st.warning(
                                        "Preview completed, but no PNG gallery "
                                        "files were found."
                                    )
                        else:
                            st.rerun()

        profile_preview_root = output_root / "profile_preview"
        profile_preview_images = gallery_png_paths(profile_preview_root)
        if profile_preview_images:
            st.markdown("### Last profile preview")
            for image_path in profile_preview_images:
                st.image(
                    str(image_path),
                    caption=str(image_path),
                    use_container_width=True,
                )

    with tab_placement:
        st.subheader("Interactive placement controls")
        scene_text = str(st.session_state[state_key])

        try:
            extent_mm = grid_extent_mm(scene_text)
            ids = object_ids(scene_text)
        except Exception as exc:
            st.error(f"{type(exc).__name__}: {exc}")
            extent_mm = [32.0, 32.0, 32.0]
            ids = []

        if not ids:
            st.info("No editable objects were found in this scene.")
        else:
            selected_geometry_object_id = st.selectbox(
                "Object to place",
                ids,
                index=0,
                key=f"placement_object::{selected_scene_id}",
            )

            try:
                selected_geometry_object = get_object(
                    scene_text,
                    selected_geometry_object_id,
                )
                controls = geometry_controls_for_object(
                    selected_geometry_object,
                    extent_mm=extent_mm,
                )
            except Exception as exc:
                st.error(f"{type(exc).__name__}: {exc}")
                controls = []

            if not controls:
                st.info(
                    "This object has no recognised geometry controls yet. "
                    "Use the Object editor or YAML tab instead."
                )
            else:
                st.caption(
                    "These controls update YAML fields directly. Validate and "
                    "render after applying changes."
                )

                updates: dict[str, object] = {}
                with st.form(
                    f"placement_editor::{selected_scene_id}::{selected_geometry_object_id}"
                ):
                    for control in controls:
                        st.markdown(f"**{control.label}**")
                        if control.help:
                            st.caption(control.help)

                        if control.kind == "point3":
                            assert isinstance(control.value, list)
                            assert isinstance(control.max_value, list)
                            values = []
                            for axis_idx, axis_name in enumerate(("i", "j", "k")):
                                values.append(
                                    st.slider(
                                        f"{control.label} {axis_name}",
                                        min_value=0.0,
                                        max_value=float(control.max_value[axis_idx]),
                                        value=float(control.value[axis_idx]),
                                        step=0.5,
                                        key=(
                                            f"{selected_scene_id}::"
                                            f"{selected_geometry_object_id}::"
                                            f"{control.path}::{axis_name}"
                                        ),
                                    )
                                )
                            updates[control.path] = values

                        elif control.kind == "positive_float":
                            max_value = (
                                float(control.max_value)
                                if control.max_value is not None
                                and not isinstance(control.max_value, list)
                                else 64.0
                            )
                            updates[control.path] = st.slider(
                                control.label,
                                min_value=0.1,
                                max_value=max_value,
                                value=float(control.value),
                                step=0.1,
                                key=(
                                    f"{selected_scene_id}::"
                                    f"{selected_geometry_object_id}::"
                                    f"{control.path}"
                                ),
                            )

                        elif control.kind == "positive_vector3":
                            assert isinstance(control.value, list)
                            max_values = (
                                control.max_value
                                if isinstance(control.max_value, list)
                                else [64.0, 64.0, 64.0]
                            )
                            values = []
                            for axis_idx, axis_name in enumerate(("i", "j", "k")):
                                values.append(
                                    st.slider(
                                        f"{control.label} {axis_name}",
                                        min_value=0.1,
                                        max_value=float(max_values[axis_idx]),
                                        value=float(control.value[axis_idx]),
                                        step=0.1,
                                        key=(
                                            f"{selected_scene_id}::"
                                            f"{selected_geometry_object_id}::"
                                            f"{control.path}::{axis_name}"
                                        ),
                                    )
                                )
                            updates[control.path] = values

                        else:
                            assert isinstance(control.value, list)
                            max_values = (
                                control.max_value
                                if isinstance(control.max_value, list)
                                else [10.0, 10.0, 10.0]
                            )
                            values = []
                            for axis_idx, axis_name in enumerate(("i", "j", "k")):
                                bound = float(max_values[axis_idx])
                                values.append(
                                    st.slider(
                                        f"{control.label} {axis_name}",
                                        min_value=-bound,
                                        max_value=bound,
                                        value=float(control.value[axis_idx]),
                                        step=0.1,
                                        key=(
                                            f"{selected_scene_id}::"
                                            f"{selected_geometry_object_id}::"
                                            f"{control.path}::{axis_name}"
                                        ),
                                    )
                                )
                            updates[control.path] = values

                    submitted = st.form_submit_button("Apply placement changes")
                    preview_submitted = st.form_submit_button(
                        "Apply and preview", type="primary"
                    )

                if submitted or preview_submitted:
                    try:
                        updated_text = update_object_geometry(
                            scene_text, selected_geometry_object_id, updates
                        )
                        st.session_state[state_key] = updated_text
                    except Exception as exc:
                        st.error(f"{type(exc).__name__}: {exc}")
                    else:
                        st.success(
                            "Updated geometry for `{selected_geometry_object_id}`."
                        )
                        if preview_submitted:
                            preview_root = output_root / "placement_preview"
                            try:
                                _ = render_preview_scene_config_text(
                                    scene_id=selected_scene_id,
                                    text=updated_text,
                                    output_root=output_root,
                                    map_name=map_name,
                                    formats=("png",),
                                    dpi=160,
                                    overwrite=True,
                                    with_colorbar=with_colorbar,
                                )
                            except Exception as exc:
                                st.error(f"{type(exc).__name__}: {exc}")
                            else:
                                st.info(
                                    "Preview rendered using map "
                                    "`{preview_result.map_name}`."
                                )
                                preview_images = gallery_png_paths(preview_root)
                                if preview_images:
                                    st.markdown("### Placement preview")
                                    for image_path in preview_images:
                                        st.image(
                                            str(image_path),
                                            caption=str(image_path),
                                            use_container_width=True,
                                        )
                                else:
                                    st.warning(
                                        "Preview completed, "
                                        "but no PNG gallery files were found"
                                    )
                else:
                    st.rerun()

        preview_root = output_root / "placement_preview"
        preview_images = gallery_png_paths(preview_root)
        if preview_images:
            st.markdown("### Last placement preview")
            for image_path in preview_images:
                st.image(
                    str(image_path), caption=str(image_path), use_container_width=True
                )

    with tab_objects:
        st.subheader("Object editor")
        scene_text = str(st.session_state[state_key])

        try:
            rows = object_summary_rows(scene_text)
            ids = object_ids(scene_text)
        except Exception as exc:
            st.error(f"{type(exc).__name__}: {exc}")
            rows = []
            ids = []

        if rows:
            st.markdown("### Objects")
            st.dataframe(rows, use_container_width=True)

        if not ids:
            st.info("No editable objects were found in this scene.")
        else:
            st.markdown("### Duplicate or delete object")
            manage_object_id = st.selectbox(
                "Object to manage",
                ids,
                index=0,
                key=f"manage_object::{selected_scene_id}",
            )

            col_duplicate, col_delete = st.columns(2)

            with col_duplicate:
                try:
                    suggested_duplicate_id = suggest_object_id(
                        scene_text,
                        f"{manage_object_id}_copy",
                    )
                    suggested_duplicate_label = suggest_next_label(scene_text)
                except Exception:
                    suggested_duplicate_id = f"{manage_object_id}_copy"
                    suggested_duplicate_label = 99

                with st.form(
                    f"duplicate_object::{selected_scene_id}::{manage_object_id}"
                ):
                    duplicate_id = st.text_input(
                        "New object ID",
                        value=suggested_duplicate_id,
                    )
                    duplicate_label = st.number_input(
                        "New label",
                        min_value=1,
                        value=int(suggested_duplicate_label),
                        step=1,
                    )
                    duplicate_offset = st.text_input(
                        "Offset mm",
                        value="[0.0, 3.0, 0.0]",
                        help=(
                            "Translation applied to common geometry fields "
                            "such as centre_mm or tube start/end coordinates."
                        ),
                    )
                    duplicate_submitted = st.form_submit_button("Duplicate object")

                if duplicate_submitted:
                    try:
                        import yaml

                        st.session_state[state_key] = duplicate_object_in_scene_text(
                            scene_text,
                            manage_object_id,
                            new_object_id=duplicate_id,
                            new_label=int(duplicate_label),
                            offset_mm=yaml.safe_load(duplicate_offset),
                        )
                    except Exception as exc:
                        st.error(f"{type(exc).__name__}: {exc}")
                    else:
                        st.success(
                            f"Duplicated `{manage_object_id}` as `{duplicate_id}`."
                        )
                        st.rerun()

            with col_delete:
                with st.form(f"delete_object::{selected_scene_id}::{manage_object_id}"):
                    st.warning(
                        "Deletion updates the YAML editor immediately. "
                        "Use Reset YAML from catalogue if you need to undo "
                        "catalogue-scene changes."
                    )
                    confirm_delete = st.checkbox(
                        f"Confirm deletion of `{manage_object_id}`",
                        value=False,
                    )
                    delete_submitted = st.form_submit_button("Delete object")

                if delete_submitted:
                    if not confirm_delete:
                        st.error("Tick the confirmation checkbox before deleting.")
                    else:
                        try:
                            st.session_state[state_key] = delete_object_from_scene_text(
                                scene_text,
                                manage_object_id,
                            )
                        except Exception as exc:
                            st.error(f"{type(exc).__name__}: {exc}")
                        else:
                            st.success(f"Deleted `{manage_object_id}`.")
                            st.rerun()

            st.markdown("### Edit object fields")
            scene_text = str(st.session_state[state_key])
            try:
                ids = object_ids(scene_text)
            except Exception:
                ids = []

            selected_object_id = st.selectbox(
                "Object",
                ids,
                index=0,
            )

            try:
                selected_object = get_object(scene_text, selected_object_id)
                editable_fields = flatten_editable_fields(selected_object)
            except Exception as exc:
                st.error(f"{type(exc).__name__}: {exc}")
                editable_fields = {}

            if editable_fields:
                st.markdown("### Editable fields")
                st.caption(
                    "Values are parsed as YAML scalars or lists. For example: "
                    "`3.0`, `target`, `true`, or `[8.0, 16.0, 16.0]`."
                )

                edits: dict[str, str] = {}
                with st.form(
                    f"object_editor::{selected_scene_id}::{selected_object_id}"
                ):
                    for field_path, value in editable_fields.items():
                        edits[field_path] = st.text_input(
                            field_path,
                            value=format_edit_value(value),
                        )

                    submitted = st.form_submit_button("Apply object edits")

                if submitted:
                    try:
                        updated_object = apply_field_edits(selected_object, edits)
                        st.session_state[state_key] = replace_object(
                            scene_text,
                            selected_object_id,
                            updated_object,
                        )
                    except Exception as exc:
                        st.error(f"{type(exc).__name__}: {exc}")
                    else:
                        st.success(
                            f"Updated object `{selected_object_id}` in the YAML editor."
                        )
                        st.rerun()

    with tab_yaml:
        st.subheader("Scene specification")
        st.text_area(
            "YAML/JSON scene config",
            key=state_key,
            height=520,
        )
        st.download_button(
            "Download current YAML text",
            data=str(st.session_state[state_key]),
            file_name=f"{selected_scene_id}.yml",
            mime="text/yaml",
        )

    with tab_validate:
        st.subheader("Validation report")

        render_during_validation = st.checkbox(
            "Also render during validation",
            value=True,
        )

        if st.button("Validate scene", type="primary"):
            try:
                if render_edited_yaml:
                    report = validate_scene_config_text(
                        scene_id=selected_scene_id,
                        text=str(st.session_state[state_key]),
                        output_root=output_root,
                        render=render_during_validation,
                    )
                else:
                    report = validate_catalogue_scene(
                        selected_scene_id,
                        render=render_during_validation,
                    )
            except Exception as exc:
                st.error(f"{type(exc).__name__}: {exc}")
            else:
                counts = report.summary_counts()
                if report.passed:
                    st.success(
                        "Validation passed "
                        f"({counts['warning']} warning(s), "
                        f"{counts['info']} info message(s))."
                    )
                else:
                    st.error(
                        "Validation failed "
                        f"({counts['error']} error(s), "
                        f"{counts['warning']} warning(s))."
                    )

                st.dataframe(report.to_rows(), use_container_width=True)

    with tab_render:
        st.subheader("Render and export")

        st.write(f"Output root: `{output_root}`")

        if st.button("Render scene", type="primary"):
            try:
                if render_edited_yaml:
                    result = render_scene_config_text(
                        scene_id=selected_scene_id,
                        text=str(st.session_state[state_key]),
                        output_root=output_root,
                        map_name=map_name,
                        formats=safe_formats,
                        dpi=int(dpi),
                        overwrite=overwrite,
                        with_colorbar=with_colorbar,
                    )
                else:
                    result = render_catalogue_scene(
                        selected_scene_id,
                        output_root=output_root,
                        map_name=map_name,
                        formats=safe_formats,
                        dpi=int(dpi),
                        overwrite=overwrite,
                        with_colorbar=with_colorbar,
                    )
            except Exception as exc:
                st.error(f"{type(exc).__name__}: {exc}")
            else:
                st.success(f"Rendered scene using map `{result.map_name}`.")
                st.write(f"Export directory: `{output_root / 'export'}`")
                st.write(f"Gallery directory: `{output_root / 'gallery'}`")

        images = gallery_png_paths(output_root)
        if images:
            st.markdown("### Gallery preview")
            for image_path in images:
                st.image(
                    str(image_path),
                    caption=str(image_path),
                    use_container_width=True,
                )
        else:
            st.info("No PNG gallery outputs found yet for this output root.")

    with tab_catalogue:
        st.subheader("Built-in catalogue")
        st.dataframe(catalogue_rows(), use_container_width=True)


if __name__ == "__main__":
    main()
