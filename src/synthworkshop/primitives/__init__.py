"""Analytic primitives for synthetic scene construction."""

from synthworkshop.primitives.curves import (
    Centreline,
    LineCurve,
    PolylineCurve,
    SinusoidalCurve,
)
from synthworkshop.primitives.frames import ReferenceFrame, build_reference_guided_frame
from synthworkshop.primitives.implicit import (
    ConeObject,
    EllipsoidObject,
    FrustumObject,
    SlabObject,
    SphereObject,
)
from synthworkshop.primitives.tubes import TubeObject

__all__ = [
    "Centreline",
    "ConeObject",
    "EllipsoidObject",
    "FrustumObject",
    "LineCurve",
    "PolylineCurve",
    "ReferenceFrame",
    "SinusoidalCurve",
    "SlabObject",
    "SphereObject",
    "TubeObject",
    "build_reference_guided_frame",
]
