from experiments.stage1_basic import run_stage1
from experiments.stage2_subsidy import run_stage2
from experiments.stage3_shortage import run_stage3
from experiments.stage4_quality_investment import run_stage4
from experiments.stage5_sensitivity import run_stage5
from experiments.stage6_mesa_abm import run_stage6
from experiments.stage7_critical_policy import run_stage7
from src.utils import ensure_dir


def main():
    output_root = "outputs_full_logit"
    ensure_dir(output_root)

    print("开始运行阶段1：基础模型实验...")
    run_stage1(output_root)
    print("阶段1完成。\n")

    print("开始运行阶段2：补贴策略实验...")
    run_stage2(output_root)
    print("阶段2完成。\n")

    print("开始运行阶段3：供给不足惩罚实验...")
    run_stage3(output_root)
    print("阶段3完成。\n")

    print("开始运行阶段4：服务质量投资实验...")
    run_stage4(output_root)
    print("阶段4完成。\n")

    print("开始运行阶段5：参数敏感性与拓展分析...")
    run_stage5(output_root)
    print("阶段5完成。\n")

    print("开始运行阶段6：Mesa Agent-Based Model 扩展实验...")
    run_stage6(output_root)
    print("阶段6完成。\n")

    print("开始运行阶段7：临界条件与策略优化补充实验...")
    run_stage7(output_root)
    print("阶段7完成。\n")

    print("全部实验完成！")
    print(f"结果已保存到文件夹：{output_root}")


if __name__ == "__main__":
    main()