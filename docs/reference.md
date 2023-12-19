# Reference

### Package `remodels`

The `remodels` package provides a set of tools and modules for quantum risk assessment. This section outlines the main components and functionalities of the `remodels` package.

#### Data Module

```{eval-rst}
.. autoclass:: remodels.data.EntsoeApi.EntsoeApi
   :members:
   :inherited-members:
```

#### Transformers Module

```{eval-rst}
.. autoclass:: remodels.transformers.BaseScaler.BaseScaler
   :members:
   :inherited-members:
```

```{eval-rst}
.. autoclass:: remodels.transformers.StandardizingScaler.StandardizingScaler
   :members:
   :inherited-members:
```

```{eval-rst}
.. autoclass:: remodels.transformers.DSTAdjuster.DSTAdjuster
   :members:
   :inherited-members:
```

##### VSTransformers SubModule

```{eval-rst}
.. autoclass:: remodels.transformers.VSTransformers.ArcsinhScaler.ArcsinhScaler
   :members:
   :inherited-members:
```

```{eval-rst}
.. autoclass:: remodels.transformers.VSTransformers.BoxCoxScaler.BoxCoxScaler
   :members:
   :inherited-members:
```

```{eval-rst}
.. autoclass:: remodels.transformers.VSTransformers.ClippingScaler.ClippingScaler
   :members:
   :inherited-members:
```

```{eval-rst}
.. autoclass:: remodels.transformers.VSTransformers.LogClippingScaler.LogClippingScaler
   :members:
   :inherited-members:
```

```{eval-rst}
.. autoclass:: remodels.transformers.VSTransformers.LogisticScaler.LogisticScaler
   :members:
   :inherited-members:
```

```{eval-rst}
.. autoclass:: remodels.transformers.VSTransformers.MLogScaler.MLogScaler
   :members:
   :inherited-members:
```

```{eval-rst}
.. autoclass:: remodels.transformers.VSTransformers.PITScaler.PITScaler
   :members:
   :inherited-members:
```

```{eval-rst}
.. autoclass:: remodels.transformers.VSTransformers.PolyScaler.PolyScaler
   :members:
   :inherited-members:
```

#### Pipeline Module

```{eval-rst}
.. autoclass:: remodels.pipelines.RePipeline.RePipeline
   :members:
   :inherited-members:
```

#### PointsModel Module

```{eval-rst}
.. autoclass:: remodels.pointsModels.PointModel.PointModel
   :members:
   :inherited-members:
```

#### QRA Models Module

```{eval-rst}
.. autoclass:: remodels.qra.qra.QRA
   :members:
   :inherited-members:
```

```{eval-rst}
.. autoclass:: remodels.qra.qrm.QRM
   :members: fit
```

```{eval-rst}
.. autoclass:: remodels.qra.lqra.LQRA
   :members: fit
```

```{eval-rst}
.. autoclass:: remodels.qra.fqra.FQRA
   :members: fit
```

```{eval-rst}
.. autoclass:: remodels.qra.fqrm.FQRM
   :members: fit
```

```{eval-rst}
.. autoclass:: remodels.qra.sfqra.sFQRA
   :members: fit
```

```{eval-rst}
.. autoclass:: remodels.qra.sfqrm.sFQRM
   :members: fit
```

```{eval-rst}
.. autoclass:: remodels.qra.sqra.SQRA
   :members: fit
```

```{eval-rst}
.. autoclass:: remodels.qra.sqrm.SQRM
   :members: fit
```

##### QR Tester SubModule

```{eval-rst}
.. autoclass:: remodels.qra.tester.qr_tester.QR_Tester
   :members:
```

```{eval-rst}
.. autoclass:: remodels.qra.tester.qr_tester.QR_TestResults
   :members: aec, ec_h, ec_mad, kupiec_test, christoffersen_test, aps, aps_extreme_quantiles
```

```{eval-rst}
.. autoclass:: remodels.qra.tester.qr_results_summary.QR_ResultsSummary
   :members:
```
