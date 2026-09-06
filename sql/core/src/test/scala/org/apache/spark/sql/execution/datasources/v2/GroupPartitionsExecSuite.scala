/*
 * Licensed to the Apache Software Foundation (ASF) under one or more
 * contributor license agreements.  See the NOTICE file distributed with
 * this work for additional information regarding copyright ownership.
 * The ASF licenses this file to You under the Apache License, Version 2.0
 * (the "License"); you may not use this file except in compliance with
 * the License.  You may obtain a copy of the License at
 *
 *    http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

package org.apache.spark.sql.execution.datasources.v2

import org.apache.spark.SparkException
import org.apache.spark.rdd.RDD
import org.apache.spark.sql.catalyst.InternalRow
import org.apache.spark.sql.catalyst.expressions.{Ascending, Attribute, AttributeReference, SortOrder, TransformExpression}
import org.apache.spark.sql.catalyst.plans.physical.{ClusteredDistribution, KeyedPartitioning, KeyedShuffleSpec, KeyReducer, Partitioning, PartitioningCollection, UnknownPartitioning}
import org.apache.spark.sql.catalyst.util.InternalRowComparableWrapper
import org.apache.spark.sql.connector.catalog.functions.{BucketFunction, BucketReducer}
import org.apache.spark.sql.execution.{DummySparkPlan, LeafExecNode, SafeForKWayMerge}
import org.apache.spark.sql.internal.SQLConf
import org.apache.spark.sql.test.SharedSparkSession
import org.apache.spark.sql.types.{IntegerType, LongType}

class GroupPartitionsExecSuite extends SharedSparkSession {

  private val exprA = AttributeReference("a", IntegerType)()
  private val exprB = AttributeReference("b", IntegerType)()
  private val exprC = AttributeReference("c", IntegerType)()
  private val exprLong = AttributeReference("l", LongType)()

  private def row(a: Int): InternalRow = InternalRow.fromSeq(Seq(a))
  private def row(a: Int, b: Int): InternalRow = InternalRow.fromSeq(Seq(a, b))

  test("SPARK-59057: the output flag reports what this node's grouping merges") {
    // Keys [(1,1), (1,2), (2,1)] projected onto position 0 give [1, 1, 2]. The first two groups
    // cover keys the child held apart, the third does not.
    val keys = Seq(row(1, 1), row(1, 2), row(2, 1))
    def gpe(joinKeyPositions: Option[Seq[Int]],
        expected: Option[Seq[(InternalRowComparableWrapper, Int)]] = None,
        distribute: Boolean = false,
        childCollapsed: Boolean = false): KeyedPartitioning = {
      val childKp = KeyedPartitioning(Seq(exprA, exprB), keys).copy(isCollapsed = childCollapsed)
      GroupPartitionsExec(DummySparkPlan(outputPartitioning = childKp), joinKeyPositions,
        expected, distributePartitions = distribute)
        .outputPartitioning.asInstanceOf[KeyedPartitioning]
    }
    def keyOf(a: Int): InternalRowComparableWrapper =
      InternalRowComparableWrapper(row(a), Seq(exprA))

    assert(!gpe(None).isCollapsed,
      "no projection, so every group covers the one key it was built from")
    assert(gpe(Some(Seq(0))).isCollapsed, "keys (1,1) and (1,2) are merged into key 1")
    assert(gpe(None, childCollapsed = true).isCollapsed, "the child's flag is sticky")

    // The keys the join agreed on decide it. Keeping key 1 keeps the merge, keeping only key 2
    // does not, and that holds however the splits of a kept key are laid out afterwards. Key 1 has
    // two child splits, which is the split count `EnsureRequirements` derives for it.
    Seq(false, true).foreach { distribute =>
      assert(gpe(Some(Seq(0)), Some(Seq(keyOf(1) -> 2, keyOf(2) -> 1)), distribute).isCollapsed,
        s"distributePartitions=$distribute: the merged key 1 survives")
      assert(!gpe(Some(Seq(0)), Some(Seq(keyOf(2) -> 1)), distribute).isCollapsed,
        s"distributePartitions=$distribute: only key 2 survives, and it merges nothing")
    }
  }

  test("SPARK-59121: a node with no reducers keeps the child's reduced key marker") {
    // This node re-reports the child's expressions, projected to `joinKeyPositions`. A reduce that
    // happened below it has to survive that, or a further join above reads the reported transform
    // as if it still described the keys.
    val reducedExpr = TransformExpression(BucketFunction, Seq(exprA), Some(12))
      .reducedTogetherWith(TransformExpression(BucketFunction, Seq(exprA), Some(8)))
    val child = DummySparkPlan(outputPartitioning =
      KeyedPartitioning(Seq(reducedExpr, exprB), Seq(row(1, 10), row(2, 20), row(1, 30))))
    val gpe = GroupPartitionsExec(child, joinKeyPositions = Some(Seq(0)))

    assert(gpe.reducers.isEmpty, "test setup: this node reduces nothing itself")
    gpe.outputPartitioning match {
      case kp: KeyedPartitioning =>
        assert(kp.expressions === Seq(reducedExpr), "the projection keeps the marked expression")
        assert(!kp.expressionsDescribeKeys)
      case other => fail(s"Expected KeyedPartitioning, got $other")
    }

    // Same for a node that reduces the other position. The position it does not reduce passes
    // through, marker and all.
    val reducingGpe = GroupPartitionsExec(child, reducers = Some(Seq(None, Some(
      KeyReducer(BucketReducer(2), TransformExpression(BucketFunction, Seq(exprB), Some(2)))))))
    reducingGpe.outputPartitioning match {
      case kp: KeyedPartitioning =>
        assert(kp.expressions.head === reducedExpr, "the unreduced position keeps its marker")
        assert(!kp.expressionsDescribeKeys)
      case other => fail(s"Expected KeyedPartitioning, got $other")
    }
  }

  test("SPARK-56241: non-coalescing passes through child ordering unchanged") {
    // Each partition has a distinct key — no coalescing happens.
    val partitionKeys = Seq(row(1), row(2), row(3))
    val childOrdering = Seq(SortOrder(exprA, Ascending))
    val child = DummySparkPlan(
      outputPartitioning = KeyedPartitioning(Seq(exprA), partitionKeys),
      outputOrdering = childOrdering)
    val gpe = GroupPartitionsExec(child)

    assert(gpe.groupedPartitions.forall(_._2.size <= 1), "expected non-coalescing")
    assert(gpe.outputOrdering === childOrdering)
  }

  test("SPARK-58324: k-way merge ordering drops sameOrderExpressions") {
    // The child ordering carries sameOrderExpressions (planner metadata). The k-way merge
    // comparator only needs the sort key, so kWayMergeOrdering keeps child/direction/nullOrdering
    // but drops sameOrderExpressions, so LazyCodeGenOrdering does not serialize them with the RDD.
    val childOrdering = Seq(SortOrder(exprA, Ascending, Seq(exprB, exprC)))
    val child = DummySparkPlan(
      outputPartitioning = KeyedPartitioning(Seq(exprA), Seq(row(1), row(2), row(1))),
      outputOrdering = childOrdering)
    val gpe = GroupPartitionsExec(child)

    assert(child.outputOrdering.head.sameOrderExpressions.nonEmpty, "test setup")
    val merged = gpe.kWayMergeOrdering
    assert(merged.map(so => (so.child, so.direction)) === Seq((exprA, Ascending)))
    assert(merged.forall(_.sameOrderExpressions.isEmpty))
  }

  test("SPARK-56241: coalescing without reducers keeps key-expression orders from child") {
    // Key 1 appears on partitions 0 and 2, causing coalescing.
    val partitionKeys = Seq(row(1), row(2), row(1))
    val child = DummySparkPlan(
      outputPartitioning = KeyedPartitioning(Seq(exprA), partitionKeys),
      outputOrdering = Seq(SortOrder(exprA, Ascending)))
    val gpe = GroupPartitionsExec(child)

    assert(!gpe.groupedPartitions.forall(_._2.size <= 1), "expected coalescing")
    // With the config disabled (default), key-expression filtering is skipped.
    assert(gpe.outputOrdering === Nil)
    // When enabled, the key-expression order is preserved through coalescing.
    withSQLConf(SQLConf.V2_BUCKETING_PRESERVE_KEY_ORDERING_ON_COALESCE_ENABLED.key -> "true") {
      val ordering = gpe.outputOrdering
      assert(ordering.length === 1)
      assert(ordering.head.child === exprA)
      assert(ordering.head.direction === Ascending)
      assert(ordering.head.sameOrderExpressions.isEmpty)
    }
  }

  test("SPARK-56241: coalescing without reducers keeps one SortOrder per key expression") {
    // Multi-key partition: key (1,10) appears on partitions 0 and 2, causing coalescing.
    val partitionKeys = Seq(row(1, 10), row(2, 20), row(1, 10))
    val child = DummySparkPlan(
      outputPartitioning = KeyedPartitioning(Seq(exprA, exprB), partitionKeys),
      outputOrdering = Seq(SortOrder(exprA, Ascending), SortOrder(exprB, Ascending)))
    val gpe = GroupPartitionsExec(child)

    assert(!gpe.groupedPartitions.forall(_._2.size <= 1), "expected coalescing")
    assert(gpe.outputOrdering === Nil)
    withSQLConf(SQLConf.V2_BUCKETING_PRESERVE_KEY_ORDERING_ON_COALESCE_ENABLED.key -> "true") {
      val ordering = gpe.outputOrdering
      assert(ordering.length === 2)
      assert(ordering.head.child === exprA)
      assert(ordering(1).child === exprB)
      assert(ordering.head.sameOrderExpressions.isEmpty)
      assert(ordering(1).sameOrderExpressions.isEmpty)
    }
  }

  test("SPARK-56241: coalescing join case preserves sameOrderExpressions from child") {
    // PartitioningCollection wraps two KeyedPartitionings (one per join side), sharing the same
    // partition keys. Key 1 coalesces partitions 0 and 2. The child (e.g. SortMergeJoinExec)
    // already carries sameOrderExpressions linking both sides' key expressions.
    val partitionKeys = Seq(row(1), row(2), row(1))
    val leftKP = KeyedPartitioning(Seq(exprA), partitionKeys)
    val rightKP = KeyedPartitioning(Seq(exprB), partitionKeys)
    val child = DummySparkPlan(
      outputPartitioning = PartitioningCollection.fromPartitionings(Seq(leftKP, rightKP)),
      outputOrdering = Seq(SortOrder(exprA, Ascending, sameOrderExpressions = Seq(exprB))))
    val gpe = GroupPartitionsExec(child)

    assert(!gpe.groupedPartitions.forall(_._2.size <= 1), "expected coalescing")
    assert(gpe.outputOrdering === Nil)
    withSQLConf(SQLConf.V2_BUCKETING_PRESERVE_KEY_ORDERING_ON_COALESCE_ENABLED.key -> "true") {
      val ordering = gpe.outputOrdering
      assert(ordering.length === 1)
      assert(ordering.head.child === exprA)
      assert(ordering.head.sameOrderExpressions === Seq(exprB))
    }
  }

  test("SPARK-56241: coalescing drops non-key sort orders from child") {
    // exprA is the partition key; exprC is a non-key sort order the child also reports
    // (e.g. a secondary sort within each partition). After coalescing, exprC ordering is lost
    // by concatenation, so only the exprA order should survive.
    val partitionKeys = Seq(row(1), row(2), row(1))
    val child = DummySparkPlan(
      outputPartitioning = KeyedPartitioning(Seq(exprA), partitionKeys),
      outputOrdering = Seq(SortOrder(exprA, Ascending), SortOrder(exprC, Ascending)))
    val gpe = GroupPartitionsExec(child)

    assert(!gpe.groupedPartitions.forall(_._2.size <= 1), "expected coalescing")
    assert(gpe.outputOrdering === Nil)
    withSQLConf(SQLConf.V2_BUCKETING_PRESERVE_KEY_ORDERING_ON_COALESCE_ENABLED.key -> "true") {
      val ordering = gpe.outputOrdering
      assert(ordering.length === 1)
      assert(ordering.head.child === exprA)
    }
  }

  test("SPARK-56241: coalescing with reducers returns empty ordering") {
    // When reducers are present, the original key expressions are not constant within the merged
    // partition, so outputOrdering falls back to the default (empty).
    val partitionKeys = Seq(row(1), row(2), row(1))
    val child = DummySparkPlan(outputPartitioning = KeyedPartitioning(Seq(exprA), partitionKeys))
    // reducers = Some(Seq(None)) - None element means identity reducer; the important thing is
    // that reducers.isDefined, which triggers the fallback.
    val gpe = GroupPartitionsExec(child, reducers = Some(Seq(None)))

    assert(!gpe.groupedPartitions.forall(_._2.size <= 1), "expected coalescing")
    assert(gpe.outputOrdering === Nil)
  }

  test("SPARK-55715: sorted merge config enabled but child not SafeForKWayMerge falls back " +
      "to key-expression ordering") {
    // DummySparkPlan does not extend SafeForKWayMerge, so childIsSafeForKWayMerge = false and
    // canUseSortedMerge = false even with enableSortedMerge = true. outputOrdering must
    // therefore fall back to key-expression filtering (not return the full child ordering).
    val partitionKeys = Seq(row(1), row(2), row(1))
    val childOrdering = Seq(SortOrder(exprA, Ascending), SortOrder(exprC, Ascending))
    val child = DummySparkPlan(
      outputPartitioning = KeyedPartitioning(Seq(exprA), partitionKeys),
      outputOrdering = childOrdering)

    assert(!GroupPartitionsExec(child).groupedPartitions.forall(_._2.size <= 1),
      "expected coalescing")
    withSQLConf(
        SQLConf.V2_BUCKETING_PRESERVE_ORDERING_ON_COALESCE_ENABLED.key -> "true",
        SQLConf.V2_BUCKETING_PRESERVE_KEY_ORDERING_ON_COALESCE_ENABLED.key -> "true") {
      // Even though enableSortedMerge = true, the child is not safe for k-way merge,
      // so only key-expression orders survive (non-key exprC is dropped).
      val ordering = GroupPartitionsExec(child, enableSortedMerge = true).outputOrdering
      assert(ordering.length === 1)
      assert(ordering.head.child === exprA)
    }
  }

  test("SPARK-55715: coalescing with enableSortedMerge = true returns full child ordering") {
    // Key 1 appears on partitions 0 and 2, causing coalescing. The child is a LeafExecNode so
    // childIsSafeForKWayMerge = true. With enableSortedMerge = true and the config enabled,
    // canUseSortedMerge = true and the full child ordering (including the non-key exprC) must be
    // returned, not just the subset of key-expression orders.
    val partitionKeys = Seq(row(1), row(2), row(1))
    val childOrdering = Seq(SortOrder(exprA, Ascending), SortOrder(exprC, Ascending))
    val child = DummyLeafSparkPlan(
      outputPartitioning = KeyedPartitioning(Seq(exprA), partitionKeys),
      outputOrdering = childOrdering)

    assert(!GroupPartitionsExec(child).groupedPartitions.forall(_._2.size <= 1),
      "expected coalescing")
    withSQLConf(SQLConf.V2_BUCKETING_PRESERVE_ORDERING_ON_COALESCE_ENABLED.key -> "true") {
      assert(GroupPartitionsExec(child).outputOrdering !== childOrdering,
        "config alone should not enable k-way merge; enableSortedMerge must be set by planner")
      assert(GroupPartitionsExec(child, enableSortedMerge = true).outputOrdering === childOrdering)
    }
    withSQLConf(
        SQLConf.V2_BUCKETING_PRESERVE_ORDERING_ON_COALESCE_ENABLED.key -> "false",
        SQLConf.V2_BUCKETING_PRESERVE_KEY_ORDERING_ON_COALESCE_ENABLED.key -> "true") {
      // Sorted-merge config disabled, key-ordering config enabled: only key-expression orders
      // survive simple concatenation (non-key exprC is dropped).
      val ordering = GroupPartitionsExec(child, enableSortedMerge = true).outputOrdering
      assert(ordering.length === 1)
      assert(ordering.head.child === exprA)
    }
  }

  test("SPARK-56549: tryEnableSortedMerge returns Some when conditions are met") {
    val partitionKeys = Seq(row(1), row(2), row(1))
    val childOrdering = Seq(SortOrder(exprA, Ascending), SortOrder(exprC, Ascending))
    val child = DummyLeafSparkPlan(
      outputPartitioning = KeyedPartitioning(Seq(exprA), partitionKeys),
      outputOrdering = childOrdering)
    val gpe = GroupPartitionsExec(child)

    withSQLConf(SQLConf.V2_BUCKETING_PRESERVE_ORDERING_ON_COALESCE_ENABLED.key -> "true") {
      val result = gpe.tryEnableSortedMerge()
      assert(result.isDefined)
      assert(result.get.enableSortedMerge)
      assert(result.get.outputOrdering === childOrdering)
    }
  }

  test("SPARK-56549: tryEnableSortedMerge returns None when config is disabled") {
    val partitionKeys = Seq(row(1), row(2), row(1))
    val childOrdering = Seq(SortOrder(exprA, Ascending))
    val child = DummyLeafSparkPlan(
      outputPartitioning = KeyedPartitioning(Seq(exprA), partitionKeys),
      outputOrdering = childOrdering)
    val gpe = GroupPartitionsExec(child)

    withSQLConf(SQLConf.V2_BUCKETING_PRESERVE_ORDERING_ON_COALESCE_ENABLED.key -> "false") {
      assert(gpe.tryEnableSortedMerge().isEmpty)
    }
  }

  test("SPARK-56549: tryEnableSortedMerge returns None when child is not SafeForKWayMerge") {
    val partitionKeys = Seq(row(1), row(2), row(1))
    val childOrdering = Seq(SortOrder(exprA, Ascending))
    // DummySparkPlan does not extend SafeForKWayMerge
    val child = DummySparkPlan(
      outputPartitioning = KeyedPartitioning(Seq(exprA), partitionKeys),
      outputOrdering = childOrdering)
    val gpe = GroupPartitionsExec(child)

    withSQLConf(SQLConf.V2_BUCKETING_PRESERVE_ORDERING_ON_COALESCE_ENABLED.key -> "true") {
      assert(gpe.tryEnableSortedMerge().isEmpty)
    }
  }

  test("SPARK-59027: createShuffleSpec subset-keys spec orders keys the same as this node's " +
      "grouping") {
    // With `allowKeysSubsetOfPartitionKeys`, `EnsureRequirements` may shuffle the other join side
    // onto the spec's projected keys while this side is re-grouped by a `GroupPartitionsExec`
    // carrying the spec's `joinKeyPositions`. The two key orders must agree (see
    // `KeyedPartitioning.groupedKeyRowOrdering`), or the sides are mis-aligned -- a planning-time
    // `PartitioningCollection` invariant failure for inner joins, silent wrong results for join
    // types that expose only one side's partitioning.
    // First-appearance order of the projected keys ([3], [1], [2]) differs from their sorted
    // order ([1], [2], [3]), so the assertion discriminates the sort each side uses.
    val partitionKeys = Seq(row(3, 30), row(1, 10), row(2, 20), row(1, 99))
    val partitioning = KeyedPartitioning(Seq(exprA, exprB), partitionKeys)

    withSQLConf(SQLConf.V2_BUCKETING_ALLOW_KEYS_SUBSET_OF_PARTITION_KEYS.key -> "true") {
      val spec = partitioning.createShuffleSpec(ClusteredDistribution(Seq(exprA)))
        .asInstanceOf[KeyedShuffleSpec]
      assert(spec.joinKeyPositions === Some(Seq(0)))

      val gpe = GroupPartitionsExec(
        DummySparkPlan(outputPartitioning = partitioning),
        joinKeyPositions = spec.joinKeyPositions)
      assert(gpe.groupedPartitions.map(_._1) === spec.partitioning.partitionKeys)
    }
  }

  test("SPARK-59234: a distributing node catches an expected partition key count " +
      "below its splits") {
    // Key 1 holds two of the child's splits. `distributePartitions` lays those splits out over the
    // count the two sides agreed on for the key, so a count of 1 does not describe this side: the
    // padding is a no-op and the key would contribute two partitions while the replicating side
    // contributes one, leaving the sides misaligned. The node has to catch that itself, since the
    // count is derived from the other side's view of this one.
    val child = DummySparkPlan(
      outputPartitioning = KeyedPartitioning(Seq(exprA), Seq(row(1), row(2), row(1))))
    def gpe(numSplitsForKey1: Int, distribute: Boolean): GroupPartitionsExec =
      GroupPartitionsExec(
        child,
        expectedPartitionKeys = Some(Seq(
          InternalRowComparableWrapper(row(1), Seq(exprA)) -> numSplitsForKey1,
          InternalRowComparableWrapper(row(2), Seq(exprA)) -> 1)),
        distributePartitions = distribute)

    val e = intercept[SparkException](gpe(1, distribute = true).groupedPartitions)
    assert(e.getCondition === "INTERNAL_ERROR")
    assert(e.getMessage.contains("expected at most 1 partition(s)"))
    assert(e.getMessage.contains("has 2 splits"))

    // A count that covers the splits is laid out as before: one split per partition, and a count
    // above the splits pads with empty partitions.
    assert(gpe(2, distribute = true).groupedPartitions.map(_._2) === Seq(Seq(0), Seq(2), Seq(1)))
    assert(gpe(3, distribute = true).groupedPartitions.map(_._2) ===
      Seq(Seq(0), Seq(2), Seq.empty, Seq(1)))

    // The replicating layout emits exactly the expected count for a key whatever its splits are,
    // so the check does not apply to it.
    assert(gpe(1, distribute = false).groupedPartitions.map(_._2) === Seq(Seq(0, 2), Seq(1)))
  }

  test("SPARK-59234: a node catches expected partition keys that are not typed like its own") {
    // The child's keys are Integer rows, so expected keys typed Long are looked up among them by an
    // equality that answers false on the types alone. Every key would miss and every partition
    // would come out empty, with no error and no rows.
    val child = DummySparkPlan(
      outputPartitioning = KeyedPartitioning(Seq(exprA), Seq(row(1), row(2))))
    val longKey = InternalRowComparableWrapper(InternalRow.fromSeq(Seq(1L)), Seq(exprLong))
    val e = intercept[SparkException] {
      GroupPartitionsExec(child, expectedPartitionKeys = Some(Seq(longKey -> 1))).groupedPartitions
    }
    assert(e.getCondition === "INTERNAL_ERROR")
    assert(e.getMessage.contains(s"expected partition keys typed [${IntegerType.simpleString}]"))
    assert(e.getMessage.contains(s"they are typed [${LongType.simpleString}]"))

    // A side that holds no key is not asked: `keyDataTypes` has no key row to report and falls back
    // to the expression types, which a reduce below would leave unrelated to the expected keys
    // (SPARK-59176). Its lookups miss either way, which is the answer for a side with nothing to
    // contribute, so it pads out to the expected partitions instead of failing.
    val emptyChild = DummySparkPlan(outputPartitioning = KeyedPartitioning(Seq(exprA), Nil))
    val emptyGpe = GroupPartitionsExec(
      emptyChild, expectedPartitionKeys = Some(Seq(longKey -> 2)), distributePartitions = true)
    assert(emptyGpe.groupedPartitions.map(_._2) === Seq(Seq.empty, Seq.empty))
  }

  test("SPARK-56549: tryEnableSortedMerge returns None when no coalescing occurs") {
    val partitionKeys = Seq(row(1), row(2), row(3))
    val childOrdering = Seq(SortOrder(exprA, Ascending))
    val child = DummyLeafSparkPlan(
      outputPartitioning = KeyedPartitioning(Seq(exprA), partitionKeys),
      outputOrdering = childOrdering)
    val gpe = GroupPartitionsExec(child)

    assert(gpe.groupedPartitions.forall(_._2.size <= 1), "expected non-coalescing")
    withSQLConf(SQLConf.V2_BUCKETING_PRESERVE_ORDERING_ON_COALESCE_ENABLED.key -> "true") {
      assert(gpe.tryEnableSortedMerge().isEmpty)
    }
  }
}

private case class DummyLeafSparkPlan(
    override val outputOrdering: Seq[SortOrder] = Nil,
    override val outputPartitioning: Partitioning = UnknownPartitioning(0)
  ) extends LeafExecNode with SafeForKWayMerge {
  override protected def doExecute(): RDD[InternalRow] =
    throw new UnsupportedOperationException
  override def output: Seq[Attribute] = Seq.empty
}
